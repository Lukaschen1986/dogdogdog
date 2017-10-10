# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: py_get_test_feats.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年07月14日 星期五 01时41分02秒
#########################################################################

import os
import numpy as np
import caffe
import cv2

def load_test_data(transformer):
    names = os.listdir('data/imgs/crop_test/')
    x_test = []
    count = 0
    for name in names:
        img = caffe.io.load_image(os.path.join('data/imgs/crop_test', name))
        transformered_imgs = transformer.preprocess('data', img)
        x_test.append(transformered_imgs.copy())
        count += 1

        if count % 1000 ==  0:
            print ('load {}/{} test files.'.format(count, len(names)))

    print ('load {}/{} test files'.format(count, len(names)))
    return x_test, names

def get_features(net, imgs, layer):
    net.blobs['data'].data[...] = imgs
    net.forward()
    feats = net.blobs[layer].data

    return feats

if __name__ == "__main__":
    caffe.set_mode_gpu()
    caffe.set_device(0)
    mu = np.array([104, 117, 123])

    folds = 5
    feat_len = 2048 * 3
    batch_size = 8
    net_model = 'ResNet50'
    data_type = ['context', 'subject', 'head']
    model_deploy = 'prototxt/' + net_model + '_deploy.prototxt'
    y_train = []
    x_train = []
    y_val = []
    x_val = []
    names = []

    for idx in range(folds):
        train_feats = []

        for l, _type in enumerate(data_type):
            model_weights = 'models/' + _type + '/Baidu_' + net_model + '_' + str(idx) + '_iter_10000.caffemodel'
            net = caffe.Net(model_deploy, model_weights, caffe.TEST)
            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_mean('data', mu)
            transformer.set_raw_scale('data', 255)
            transformer.set_channel_swap('data', (2, 1, 0))
            net.blobs['data'].reshape(batch_size, 3, 224, 224)

            if _type == 'c' and idx == 0:
                x_train, names = load_test_data(transformer)
                x_train = np.array(x_train)
                train_feats = np.zeros((x_train.shape[0], feat_len + 1))

            x_train_feats = np.zeros((x_train.shape[0], feat_len / len(data_type)))
            k = 0
            while k + batch_size < x_train.shape[0]:
                imgs = x_train[k:k + batch_size, :, :, :]
                x_train_feats[k:k + batch_size] = get_features(net, imgs, 'pool5')
                k += batch_size
            if k < x_train.shape[0]:
                net.blobs['data'].reshape[x_train.shape[0] - k, 3, 224, 224]
                imgs = x_train[k:, :, :, :]
                x_train_feats[k:] = get_features(net, imgs, 'pool5')

            if l == 0:
                train_feats = x_train_feats
            else:
                train_feats = np.hstack((train_feats, x_train_feats))

            if l == len(data_type) - 1:
                train_feats = np.hstack((train_feats, np.array(names)))

        np.save('feats/test' + str(idx) + '.npy', train_feats)
