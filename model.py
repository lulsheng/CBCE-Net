import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

import tensorflow as tf

from tensorflow_deeplab_resnet.deeplab_resnet.model import DeepLabResNetModel as deeplab101


from util.processing_tools import *
from util import loss

class CBCENet(object):

    def __init__(self, batch_size = 1,
                       num_steps = 7,
                       H= 320,
                       W =320,
                       vf_h = 40,
                       vf_w = 40,
                       vf_dim = 2048,
                       v_emb_dim = 1000,
                       w_emb_dim = 1000,
                       atrous_dim = 512,
                       mlp_dim = 500,
                       rnn_size = 1000,
                       start_lr = 0.00025,
                       lr_decay_step = 800000,
                       lr_decay_rate = 1.0,
                       num_rnn_layers=1,
                       emb_name = 'pad',
                       phrase_num = 4,
                       mode = 'train',
                       weight_decay = 0.0005,
                       optimizer = 'adam',
                       weights = 'deeplab',
                       ):
        self.batch_size = batch_size
        self. num_steps = num_steps
        self. H = H
        self. W = W
        self.vf_h = vf_h
        self.vf_w = vf_w
        self.vf_dim = vf_dim
        self.v_emb_dim = v_emb_dim
        self.w_emb_dim = w_emb_dim
        self.mlp_dim = mlp_dim
        self.emb_name = emb_name
        self.phrase_num = phrase_num
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.weight_decay = weight_decay
        self.start_lr = start_lr
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.optimizer = optimizer
        self.mode = mode
        self.weights = weights
        self.atrous_dim = atrous_dim
        self.conv_dim = 256

        self.words = tf.placeholder(tf.int32, [self.batch_size, self.phrase_num, self.num_steps])
        self.im = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 3])
        self.target_fine = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 1])
        self.valid_idx = tf.placeholder(tf.int32, [self.batch_size, 1])

        resmodel = deeplab101({'data': self.im}, is_training=False)
        self.visual_feat_c5 = resmodel.layers['res5c_relu']  # 1, 40, 40, 2048
        self.visual_feat_c4 = resmodel.layers['res4b22_relu']  # 1, 40, 40, 1024
        self.visual_feat_c3 = resmodel.layers['res3b3_relu'] # 1, 40, 40, 512
        self.spatial= tf.convert_to_tensor(generate_spatial_batch(self.batch_size, vf_h, vf_w))  # 1, 40, 40, 8

       # Glove Embedding
        glove_np = np.load('./glove_pre/{}_emb.npy'.format(self.emb_name))  # 'emb_name': 'pad'
        print("Loaded embedding npy at data/{}_emb.npy".format(self.emb_name))
        self.glove = tf.convert_to_tensor(glove_np, tf.float32)  # [vocab_size, 400]

        with tf.variable_scope("text_objseg"):
            self.build_graph()
            if self.mode == 'eval':
                return
            self.train_op()


    def build_graph(self):

        embedding_mat = tf.Variable(self.glove)
        embedded_seq = tf.nn.embedding_lookup(embedding_mat, tf.transpose(self.words))  # [num_step, batch_size, glove_emb]
        print("Build Glove Embedding.")

        rnn_cell_basic = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=False)   # rnn_size: 1000
        cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell_basic] * self.num_rnn_layers, state_is_tuple=False)   # 1 layer LSTM

        state = cell.zero_state(self.batch_size, tf.float32)
        state_shape = state.get_shape().as_list()
        state_shape[0] = self.batch_size   # 1
        state.set_shape(state_shape)
        h_a = tf.zeros([self.batch_size, self.rnn_size])
        words_feat_list = []
        h_a_list = []

        def f1():
            return state, h_a   # 1, 1000

        def f2():
            # Embed words
            w_emb = embedded_seq[n, :, :]
            with tf.variable_scope("LSTM"):
                h_w, state_w_ret = cell(w_emb, state)
            return state_w_ret, h_w

        with tf.variable_scope("RNN"):
            for i in range(self.phrase_num):
                phrase = tf.expand_dims(self.words[0][i], 0 )   # phrase: 1, 7
                embedded_seq = tf.nn.embedding_lookup(embedding_mat, tf.transpose(phrase))   # embedded_seq: 7,1,300
                print("Build Glvoe Embedding.")
                for n in range(self.num_steps):
                    if n > 0:
                        tf.compat.v1.get_variable_scope().reuse_variables()
                    state_w, h_a = tf.cond(tf.equal(phrase[0,n], tf.constant(0)), lambda: f1(), lambda: f2())   # h_a 是lstm最后的输出
                h_a_list.append(h_a)   # h_a: 1,1000, h_a_list: 5,1000
        lang_feat = tf.concat(h_a_list, 0)   # 5, 1000
        lang_feat = tf.nn.l2_normalize(tf.reduce_max(lang_feat, axis=0, keep_dims=True))   #* max
        lang_feat = tf.reshape(lang_feat, [1, 1, 1, 1000])


        visual_feat_c5 = self._conv("c5", self.visual_feat_c5, 1, self.vf_dim, self.v_emb_dim, [1, 1, 1, 1])
        visual_feat_c5 = tf.nn.l2_normalize(visual_feat_c5, 3)  #1， 40，40， 1000
        visual_feat_c4 = self._conv("c4", self.visual_feat_c4, 1, 1024, self.v_emb_dim, [1, 1, 1, 1])
        visual_feat_c4 = tf.nn.l2_normalize(visual_feat_c4, 3)   # 1， 40，40，1000
        visual_feat_c3 = self._conv("c3", self.visual_feat_c3, 1, 512, self.v_emb_dim, [1, 1, 1, 1])
        visual_feat_c3 = tf.nn.l2_normalize(visual_feat_c3, 3) # 1， 40， 40， 1000

        fusion_5 = self.build_lang2vis(visual_feat_c5, lang_feat, self.spatial, level='c5')
        fusion_4 = self.build_lang2vis(visual_feat_c4, lang_feat, self.spatial, level='c4')
        fusion_3 = self.build_lang2vis(visual_feat_c3, lang_feat, self.spatial, level='c3')

        # For multi-level losses
        score_c5 = self._conv("score_c5", fusion_5, 3, self.mlp_dim, 1, [1, 1, 1, 1])
        self.up_c5 = tf.compat.v1.image.resize_bilinear(score_c5, [self.H, self.W])
        score_c4 = self._conv("score_c4", fusion_4, 3, self.mlp_dim, 1, [1, 1, 1, 1])
        self.up_c4 = tf.compat.v1.image.resize_bilinear(score_c4, [self.H, self.W])
        score_c3 = self._conv("score_c3", fusion_3, 3, self.mlp_dim, 1, [1, 1, 1, 1])
        self.up_c3 = tf.compat.v1.image.resize_bilinear(score_c3, [self.H, self.W])

        feat5_exg = self.CIM('feat_c5', lang_feat, fusion_5, fusion_3, fusion_4, level='c5')  # 1, 40, 40, 256
        feat5_exg = tf.nn.l2_normalize(feat5_exg, 3)
        feat4_exg = self.CIM('feat_c4', lang_feat, fusion_4, fusion_5, fusion_3, level='c4')   # 1, 40, 40, 256
        feat4_exg = tf.nn.l2_normalize(feat4_exg, 3)
        feat3_exg = self.CIM('feat_c3', lang_feat, fusion_3, fusion_5, fusion_4, level='c3')    # 1, 40, 40, 256
        feat3_exg = tf.nn.l2_normalize(feat3_exg, 3)

        feat5_exg_2 = self.CIM('feat_c5_2', lang_feat, feat5_exg, feat3_exg, feat4_exg, level='c5_2')  # 1, 40, 40, 256
        feat5_exg_2 = tf.nn.l2_normalize(feat5_exg_2, 3)
        feat4_exg_2 = self.CIM('feat_c4', lang_feat, feat4_exg, feat3_exg, feat5_exg, level='c4_2')   # 1, 40, 40, 256
        feat4_exg_2 = tf.nn.l2_normalize(feat4_exg_2, 3)
        feat3_exg_2 = self.CIM('feat_c3', lang_feat, feat3_exg, feat4_exg, feat5_exg, level='c3_2')    # 1, 40, 40, 256
        feat3_exg_2 = tf.nn.l2_normalize(feat3_exg_2, 3)


        self.feat = tf.concat([feat3_exg_2 , feat4_exg_2 , feat5_exg_2], 3) # 1, 40, 40, 1500
        conv2 = self._conv("conv2", self.feat, 1, self.feat.shape[3], 1000, [1, 1, 1, 1])  # 1, 40, 40, 1000
        conv2 = tf.nn.relu(conv2)
        # ASPP
        atrous_C_1 = self._atrous_conv("atrous_C_1", conv2, 3, conv2.shape[3], self.atrous_dim, 1)
        atrous_C_3 = self._atrous_conv("atrous_C_3", conv2, 3, conv2.shape[3], self.atrous_dim, 3)
        atrous_C_5 = self._atrous_conv("atrous_C_5", conv2, 3, conv2.shape[3], self.atrous_dim, 5)
        atrous_C_7 = self._atrous_conv("atrous_C_7", conv2, 3, conv2.shape[3], self.atrous_dim, 7)
        atrous_con = tf.concat([atrous_C_1, atrous_C_3, atrous_C_5, atrous_C_7, conv2], 3)
        final_out = self._conv("conv_final_out", atrous_con, 1, atrous_con.shape[3], 1 , [1, 1, 1, 1])
        self.pred = final_out
        self.up =  tf.compat.v1.image.resize_bilinear(self.pred, [self.H, self.W])
        self.sigm = tf.sigmoid(self.up)


    def build_lang2vis(self, visual_feat, lang_feat, spatial, level=""):
        vis_la_sp = self.linear_fuse(lang_feat, spatial, visual_feat, level=level)   # 1, 40, 40, 1000
        lang_vis_feat = tf.tile(lang_feat, [1, self.vf_h, self.vf_w, 1])  # [B, H, W, C]
        feat_all = tf.concat([vis_la_sp, lang_vis_feat, spatial], 3)
        # Feature fusion
        fusion = self._conv("fusion_{}".format(level), feat_all, 1,
                            self.v_emb_dim * 2 + 8,
                            self.mlp_dim, [1, 1, 1, 1])
        fusion = tf.nn.relu(fusion)
        return fusion


    def CIM(self, name, lang_feat, visual_feat1, visual_feat2, visual_feat3, level):
        with tf.variable_scope(name):
            l1= self.VGLM('vglm', lang_feat, visual_feat1, level)   #
            v1 = self.LGVM_2('lgvm2', l1, visual_feat2, visual_feat3, level) # 1, 40, 40, 500
            return v1


    def LGVM_2(self, name, lang_feat, vis_feat1, vis_feat2, level):
        with tf.variable_scope(name):
            feat1 = self.filter(vis_feat1, lang_feat, level + '_f1')
            feat2 = self.filter(vis_feat2, lang_feat, level + '_f2')
            out = vis_feat1 + feat1 + feat2
            return out

    def filter(self, feat, lang, level=""):

        lang_feat = self._conv("lang_feat_{}".format(level),
                                     lang, 1, self.mlp_dim, self.mlp_dim, [1, 1, 1, 1])  # [B, 1, 1, C]
        lang_feat = tf.sigmoid(lang_feat)
        feat_trans = self._conv("trans_feat_{}".format(level),
                                feat, 1, self.mlp_dim, self.mlp_dim, [1, 1, 1, 1])  # [B, H, W, C]
        feat_trans = tf.nn.relu(feat_trans)
        # use lang feat as a channel filter
        feat_trans = feat_trans * lang_feat # [B, H, W, C]
        return feat_trans

    def VGLM(self, name, lang_feat, vis_feat, level=""):

        with tf.variable_scope(name):
            feat_key = self._conv("vis_key_{}".format(level), vis_feat, 1, self.mlp_dim, self.mlp_dim, [1, 1, 1, 1])  # 1, 40, 40, 500
            feat_key = tf.reshape(feat_key, [self.batch_size, self.vf_h * self.vf_w, self.mlp_dim])  # [B, HW, C]  # 1, 1600, 500
            lang_query = self._conv("lang_query_{}".format(level), lang_feat, 1, self.rnn_size, self.mlp_dim, [1, 1, 1, 1]) # 1, 1, 1, 500
            lang_query = tf.reshape(lang_query, [self.batch_size, 1, self.mlp_dim])  # [B, 1, C]  # 1, 1, 500

            attn_map = tf.matmul(feat_key, lang_query, transpose_b=True)  # [B, HW, 1]   1, 1600, 1
            # Normalization for affinity matrix
            attn_map = tf.divide(attn_map, self.mlp_dim ** 0.5)
            attn_map = tf.nn.softmax(attn_map, axis=1)
            # attn_map: [B, HW, 1]

            feat_reshaped = tf.reshape(vis_feat, [self.batch_size, self.vf_h * self.vf_w, self.mlp_dim])   # 1, 1600, 500
            # feat_reshaped: [B, HW, C]
            # Adaptive global average pooling
            gv_pooled = tf.matmul(attn_map, feat_reshaped, transpose_a=True)  # [B, 1, C]
            gv_pooled = tf.reshape(gv_pooled, [self.batch_size, 1, 1, self.mlp_dim])  # [B, 1, 1, C]

            gv_lang = tf.concat([gv_pooled, lang_feat], 3)  # [B, 1, 1, 3C]
            gv_lang = self._conv("gv_lang_{}".format(level), gv_lang, 1, self.mlp_dim + self.rnn_size, self.rnn_size,
                                [1, 1, 1, 1])  # [B, 1, 1, C]  #
            gv_lang = tf.nn.l2_normalize(gv_lang)

            return gv_lang

    def linear_fuse_head(self, lang_feat, spatial_feat, visual_feat, level=''):    # lang_feat: 1, 1, 1, 1000
        # visual feature transform
        vis = tf.concat([visual_feat, spatial_feat], 3)   # [B, H, W, C+8]
        vis = self._conv("vis_{}".format(level), vis, 1,
                               self.v_emb_dim+8, self.v_emb_dim, [1, 1, 1, 1])     # 1, 40, 40, 1000
        vis = tf.nn.tanh(vis)  # [B, H, W, C]  1, 40, 40, 1000

        # lang feature transform
        lang = self._conv("lang_{}".format(level), lang_feat,
                                1, self.rnn_size, self.v_emb_dim, [1, 1, 1, 1])   # 1, 1, 1, 1000

        lang = tf.nn.tanh(lang)  # [B, 1, 1, C]  1, 1, 1, 1000

        fusion_feat = vis * lang  # [B, H, W, C]
        return fusion_feat

    def linear_fuse(self, lang_feat, spatial_feat, visual_feat, level=''):
        # fuse language feature and visual feature
        # lang_feat: [B, 1, 1, C], visual_feat: [B, H, W, C], spatial_feat: [B, H, W, 8]
        # output: [B, H, W, C']
        head1 = self.linear_fuse_head(lang_feat, spatial_feat, visual_feat, '{}_head1'.format(level))
        head2 = self.linear_fuse_head(lang_feat, spatial_feat, visual_feat, '{}_head2'.format(level))
        head3 = self.linear_fuse_head(lang_feat, spatial_feat, visual_feat, '{}_head3'.format(level))
        head4 = self.linear_fuse_head(lang_feat, spatial_feat, visual_feat, '{}_head4'.format(level))
        head5 = self.linear_fuse_head(lang_feat, spatial_feat, visual_feat, '{}_head5'.format(level))

        fused_feats = tf.stack([head1, head2, head3, head4, head5], axis=4)  # [B, H, W, C, 5] 1, 40, 40, 1000, 5
        fused_feats = tf.reduce_sum(fused_feats, 4)  # [B, H, W, C]
        fused_feats = tf.nn.tanh(fused_feats)
        fused_feats = tf.nn.l2_normalize(fused_feats, 3)

        return fused_feats

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.conv2d(x, w, strides, padding='SAME') + b

    def _atrous_conv(self, name, x, filter_size, in_filters, out_filters, rate):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
                initializer=tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.atrous_conv2d(x, w, rate=rate, padding='SAME') + b

    def train_op(self):
        tvars = [var for var in tf.trainable_variables() if var.op.name.startswith('text_objseg')]
        reg_var_list = [var for var in tvars if var.op.name.find(r'DW') > 0 or var.name[-9:-2] == 'weights']
        print('Collecting variables for regularization:')
        for var in reg_var_list: print('\t %s' % var.name)
        print('-'*20 + 'Done.')

        print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        # define loss
        self.target = tf.compat.v1.image.resize_bilinear(self.target_fine, [self.vf_h, self.vf_w])
        self.cls_loss = loss.weighed_logistic_loss(self.up, self.target_fine, 1, 1)
        self.cls_loss_c5 = loss.weighed_logistic_loss(self.up_c5, self.target_fine, 1, 1)
        self.cls_loss_c4 = loss.weighed_logistic_loss(self.up_c4, self.target_fine, 1, 1)
        self.cls_loss_c3 = loss.weighed_logistic_loss(self.up_c3, self.target_fine, 1, 1)
        self.cls_loss_all = 0.7 * self.cls_loss + 0.1 * self.cls_loss_c5 + 0.1 * self.cls_loss_c4 + 0.1 * self.cls_loss_c3
        # self.cls_loss_all = self.cls_loss
        self.reg_loss = loss.l2_regularization_loss(reg_var_list, self.weight_decay)
        self.cost = self.cls_loss_all + self.reg_loss

        # learning rate
        lr = tf.Variable(0.0, trainable=False)
        self.learning_rate = tf.compat.v1.train.polynomial_decay(self.start_lr, lr, self.lr_decay_step, end_learning_rate=0.00001, power=0.9)

        # optimizer
        if self.optimizer == 'adam':
            optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        else:
            raise ValueError("Unknown optimizer type %s!" % self.optimizer)


        # learning rate nultiplier
        grads_and_vars = optimizer.compute_gradients(self.cost, var_list=tvars)
        var_lr_mult = {}
        for var in tvars:
            if var.op.name.find(r'biases') > 0:
                var_lr_mult[var] = 2.0
            elif var.name.startswith('res5') or var.name.startswith('res4') or var.name.startswith('res3'):
                var_lr_mult[var] = 1.0
            else:
                var_lr_mult[var] = 1.0
        print('Variable learning rate multiplication:')
        for var in tvars:
            print('\t%s: %f' % (var.name, var_lr_mult[var]))
        print('Done')
        grads_and_vars = [((g if var_lr_mult[v] ==1 else tf.multiply(var_lr_mult[v], g)), v) for g,v in grads_and_vars]

        # training step
        self.train_step = optimizer.apply_gradients(grads_and_vars, global_step=lr)