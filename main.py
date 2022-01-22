from __future__ import division

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import tensorflow as tf
import skimage
from skimage import io as sio
import time
import cv2
from pydensecrf import densecrf

from util import data_reader
from util.processing_tools import *
from util import im_processing, eval_tools
from model import CBCENet


def train(modelname, max_iter, snapshot, dataset, weights, setname, mu, lr, bs, tfmodel_folder, re_iter):

    iters_per_log = 500
    data_folder = '/home/lls/project/pad_dataset/pad_phrase/ablation/train_phrase_4_batch'
    data_prefix = 'pad_phrase_train_phrase_4' #

    tfmodel_folder = '/home/lls/project/PADSeg/save_models/ablation/train_phrase_4'
    snapshot_file = os.path.join(tfmodel_folder, dataset + '_' + weights + '_' + modelname + '_iter_%d.tfmodel')

    if not os.path.isdir(tfmodel_folder):
        os.makedirs(tfmodel_folder)


    cls_loss_avg = 0
    avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0, 0, 0
    decay = 0.9


    model = CBCENet(mode='train', start_lr=lr, batch_size=bs)

    if re_iter is None:
        pretrained_model = '/home/lls/project/PADSeg/models/deeplab_resnet_init.ckpt'
        load_var = {var.op.name: var for var in tf.compat.v1.global_variables()
                   if var.name.startswith('res') or var.name.startswith('bn') or var.name.startswith('conv1')}
        snapshot_loader = tf.compat.v1.train.Saver(load_var)
        snapshot_saver = tf.compat.v1.train.Saver(max_to_keep = 1000)

    else:
        print('resume from %d' % re_iter)
        pretrained_model = os.path.join(tfmodel_folder, dataset + '_' + weights + '_' + modelname  + '_iter_' + str(re_iter) + '.tfmodel')
        snapshot_loader = tf.compat.v1.train.Saver()
        snapshot_saver = tf.compat.v1.train.Saver(max_to_keep = 1000)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    snapshot_loader.restore(sess, pretrained_model)

    im_h, im_w, num_steps , phrase_num = model.H, model.W, model.num_steps, model.phrase_num
    # text_batch = np.zeros((bs , num_steps), dtype=np.float32)
    text_batch = np.zeros((bs , phrase_num, num_steps), dtype=np.float32)
    image_batch = np.zeros((bs, im_h, im_w, 3), dtype=np.float32)
    mask_batch = np.zeros((bs, im_h, im_w, 1), dtype=np.float32)
    valid_idx_batch = np.zeros((bs, 1), dtype=np.int32)

    reader = data_reader.DataReader(data_folder, data_prefix)

    for n_iter in range(max_iter):
        for n_batch in range(bs):
            batch = reader.read_batch(is_log = (n_batch==0 and n_iter%iters_per_log==0))
            text = batch['text_batch']
            im = batch['im_batch'].astype(np.float32)
            mask = np.expand_dims(batch['mask_batch'].astype(np.float32), axis=2)

            im = im[:, :, ::-1]
            im -= mu

            text_batch[n_batch, ...] = text
            image_batch[n_batch, ...] = im
            mask_batch[n_batch, ...] = mask

            # for idx in range(text.shape[0]):
            #     if text[idx] != 0:
            #         valid_idx_batch[n_batch, :] = idx
            #         break

        _, cls_loss_val, lr_val, scores_val, label_val, up_val   = sess.run([model.train_step,
                model.cls_loss,
                model.learning_rate,
                model.pred,
                model.target,
                model.up
                ],
                feed_dict={
                    model.words: text_batch,
                    model.im: image_batch,
                    model.target_fine: mask_batch,
                    model.valid_idx: valid_idx_batch
                })

        cls_loss_avg = decay*cls_loss_avg + (1-decay)*cls_loss_val

        # Accuracy
        accuracy_all, accuracy_pos, accuracy_neg = compute_accuracy(scores_val, label_val)
        avg_accuracy_all = decay*avg_accuracy_all + (1-decay)*accuracy_all
        avg_accuracy_pos = decay*avg_accuracy_pos + (1-decay)*accuracy_pos
        avg_accuracy_neg = decay*avg_accuracy_neg + (1-decay)*accuracy_neg

        tmp = './tmp'

        if n_iter%iters_per_log==0:
            print('iter = %d, loss (cur) = %f, loss (avg) = %f, lr = %f'
                    % (n_iter, cls_loss_val, cls_loss_avg, lr_val))
            #print('iter = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
            #        % (n_iter, accuracy_all, accuracy_pos, accuracy_neg))
            print('iter = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
                    % (n_iter, avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg))

        if  n_iter%1000==0:
            up_val = np.squeeze(up_val)
            pred_raw = (up_val >= 1e-9).astype(np.float32)
            predicts = im_processing.resize_and_crop(pred_raw, mask.shape[0],
                                                    mask.shape[1])
            phrases = str(n_iter)
            mask_1 = np.squeeze(mask)
            visualize_seg(im, mask_1, predicts, phrases, tmp)


        # Save snapshot
        if (n_iter+1) % snapshot == 0 or (n_iter+1) >= max_iter:
            snapshot_saver.save(sess, snapshot_file % (n_iter+1))
            print('snapshot saved to ' + snapshot_file % (n_iter+1))

    print('Optimization done.')

def test(modelname, visualize,iter, dataset,  weights, setname, dcrf, mu, tfmodel_folder, save_results):
    data_folder = './dataset/val/'
    data_prefix = 'val'
    tfmodel_folder = './save_models'  # the pre-trained models

    pretrained_model = os.path.join(tfmodel_folder, dataset + '_deeplab_CBCE_iter_' + str(iter) + '.tfmodel')

    H, W = 320, 320


    model = CBCENet(H=H, W=W, mode='eval',weights=weights)

    # Load pretrained model
    snapshot_restorer = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    snapshot_restorer.restore(sess, pretrained_model)
    reader = data_reader.DataReader(data_folder, data_prefix, shuffle=False)

    NN = reader.num_batch
    print('test in', dataset, setname)

    for n_iter in range(reader.num_batch):

        if n_iter % (NN//50) == 0:
            if n_iter/(NN//50)%5 == 0:
                sys.stdout.write(str(n_iter/(NN//50)//5))
            else:
                sys.stdout.write('.')
            sys.stdout.flush()

        batch = reader.read_batch(is_log = False)
        text = batch['text_batch']
        im = batch['im_batch']
        mask = batch['mask_batch'].astype(np.float32)
        aff = str(batch['affordance'][0])
        obj = str(batch['obj'][0])
        img_name = str(batch['img_name'][0]).split('.')[0]


        proc_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, H, W))
        proc_im_ = proc_im.astype(np.float32)
        proc_im_ = proc_im_[:,:,::-1]
        proc_im_ -= mu

        scores_val, up_val, sigm_val = sess.run([model.pred, model.up, model.sigm],
            feed_dict={
                model.words: np.expand_dims(text, axis=0),
                model.im: np.expand_dims(proc_im_, axis=0)
            })

        up_val = np.squeeze(up_val)
        up_val = sigmoid(up_val)
        predicts = 255 * up_val
        predicts = im_processing.resize_and_crop(predicts, mask.shape[0], mask.shape[1])


        if dcrf:
            # Dense CRF post-processing
            sigm_val = np.squeeze(sigm_val)
            d = densecrf.DenseCRF2D(W, H, 2)
            U = np.expand_dims(-np.log(sigm_val), axis=0)
            U_ = np.expand_dims(-np.log(1 - sigm_val), axis=0)
            unary = np.concatenate((U_, U), axis=0)
            unary = unary.reshape((2, -1))
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=proc_im, compat=10)
            Q = d.inference(5)
            pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)
            pred_raw_dcrf = 255 * pred_raw_dcrf
            predicts_dcrf = im_processing.resize_and_crop(pred_raw_dcrf, mask.shape[0], mask.shape[1])

        if save_results:
            if dcrf:
                vis_dir = "./visualize/result_{}".format(iter)
                dir_name = os.path.join(vis_dir, aff + '/' + obj)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                file_path = str(dir_name) + '/' + img_name + '.png'
                cv2.imwrite(file_path, predicts_dcrf)
            else:
                vis_dir = "./save_results/pad_{}".format(iter)
                dir_name = os.path.join(vis_dir, aff + '/' + obj)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                file_path = str(dir_name) + '/' + img_name + '.png'
                cv2.imwrite(file_path, predicts)

        if visualize:
            vis_dir = "./visualize/{}".format(iter)
            dirs = aff + '/' + obj + '/' + img_name

            visualize_seg(im, mask, predicts, dirs, vis_dir)
            if dcrf:
                visualize_seg(im, mask, predicts_dcrf, dirs, vis_dir)


def visualize_seg(im, mask, predicts, sent, vis_dir):
    # print("visualizing...")

    sent_dir = os.path.join(vis_dir, sent)
    if not os.path.exists(sent_dir):
        os.makedirs(sent_dir)

    # Ignore sio warnings of low-contrast image.
    im_seg = im / 2
    im_seg[:, :, 0] += predicts.astype('uint8') * 100
    im_seg = im_seg.astype('uint8')
    sio.imsave(os.path.join(sent_dir, "pred.png"), im_seg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type = str, default = '3')
    parser.add_argument('-m', type=str, default = 'test')
    parser.add_argument('-n', type=str, default = 'PADSeg')
    parser.add_argument('-s', type=int, default = 75141)
    parser.add_argument('-d', type = str, default = 'pad')
    parser.add_argument('-i', type=int, default = 75141)  # 100 epoch
    parser.add_argument('-bs', type = int, default = 1) # batch size
    parser.add_argument('-re', type = int, default = None)
    parser.add_argument('-t', type = str, default='val') # 'train' 'trainval' 'val' 'test' 'testA' 'testB'
    parser.add_argument('-lr', type = float, default = 0.00025) # start learning rate
    parser.add_argument('-w', type = str, default = 'deeplab') # 'resnet' 'deeplab'
    parser.add_argument('-sfolder', type = str)
    parser.add_argument('-c', default = False, action = 'store_true') # whether or not apply DenseCRF
    parser.add_argument('-v', default=False, action='store_true')  # if
    parser.add_argument('-save', default=False, action='store_true')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    mu = np.array((104.00698793, 116.66876762, 122.67891434))

    if args.m == 'train':
        train(
            modelname = args.n,
            snapshot = args.s,
            max_iter = args.i,
            bs = args.bs,
            re_iter = args.re,
            mu = mu,
            dataset = args.d,
            setname = args.t,
            lr = args.lr,
            weights = args.w,
            tfmodel_folder = args.sfolder,
        )
    elif args.m == 'test':
        test(modelname = args.n,
            visualize=args.v,
            iter = args.i,
            dataset = args.d,
            weights = args.w,
            setname = args.t,
            dcrf = args.c,
            mu = mu,
            tfmodel_folder = args.sfolder,
            save_results= args.save)