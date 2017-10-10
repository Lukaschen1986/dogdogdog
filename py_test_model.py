# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: py_test_model.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年07月03日 星期一 15时06分42秒
#########################################################################

import os
import numpy as np
import caffe

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

def predict_label(net, imgs):
    net.blobs['data'].data[...] = imgs
    output = net.forward()
    output_prob = output['prob']

    return output_prob

def merge_several_prob(probs, folds):
    a = probs[0]
    a = np.array(a)
    for idx in range(1, folds):
        a += np.array(probs[idx])

    a /= folds
    return a.argmax(axis = 1)

if __name__ == '__main__':

    caffe.set_mode_gpu()
    caffe.set_device(0)
    mu = np.array([104, 117, 123])

    folds = 5
    class_num = 100
    batch_size = 8
    net_model = 'ResNet101'
    yfull_test = []
    pred_list = []
    x_test = []
    y_test = []
    names = []
    for idx in range(folds):

        model_deploy = 'prototxt/' + net_model + '_deploy.prototxt'
        model_weights = 'models/Baidu_' + net_model + '_' + str(idx) + '_iter_10000.caffemodel'
        net = caffe.Net(model_deploy, model_weights, caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', mu)
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
        net.blobs['data'].reshape(batch_size, 3, 224, 224)

        if idx == 0:
            x_test, names = load_test_data(transformer)
            x_test = np.array(x_test)

        pred_prob = np.zeros((x_test.shape[0], class_num))
        k = 0
        while k + batch_size < x_test.shape[0]:
            imgs = x_test[k:k + batch_size, :, :, :]
            pred_prob[k:k + batch_size] = predict_label(net, imgs)
            k += batch_size
        if k < x_test.shape[0]:
            net.blobs['data'].reshape(x_test.shape[0] - k, 3, 224, 224)
            imgs = x_test[k:, :, :, :]
            pred_prob[k:] = predict_label(net, imgs)

        yfull_test.append(pred_prob)
        pred_list.append(pred_prob.argmax(axis = 1).tolist())
    #for idx in range(10):

    #    model_deploy = 'prototxt/' + net_model + '_deploy.prototxt'
    #    model_weights = 'models/Baidu_' + net_model + '_' + str(idx) + '_iter_10500.caffemodel'
    #    net = caffe.Net(model_deploy, model_weights, caffe.TEST)
    #    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    #    transformer.set_transpose('data', (2, 0, 1))
    #    transformer.set_mean('data', mu)
    #    transformer.set_raw_scale('data', 255)
    #    transformer.set_channel_swap('data', (2, 1, 0))
    #    net.blobs['data'].reshape(batch_size, 3, 224, 224)

    #    pred_prob = np.zeros((x_test.shape[0], class_num))
    #    k = 0
    #    while k + batch_size < x_test.shape[0]:
    #        imgs = x_test[k:k + batch_size, :, :, :]
    #        pred_prob[k:k + batch_size] = predict_label(net, imgs)
    #        k += batch_size
    #    if k < x_test.shape[0]:
    #        net.blobs['data'].reshape(x_test.shape[0] - k, 3, 224, 224)
    #        imgs = x_test[k:, :, :, :]
    #        pred_prob[k:] = predict_label(net, imgs)

    #    yfull_test.append(pred_prob)

    net_model = 'VGG16'
    batch_size = 32
    for idx in range(5):

        model_deploy = 'prototxt/' + net_model + '_deploy.prototxt'
        model_weights = 'models/Baidu_' + net_model + '_' + str(idx) + '_iter_10000.caffemodel'
        net = caffe.Net(model_deploy, model_weights, caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', mu)
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
        net.blobs['data'].reshape(batch_size, 3, 224, 224)

        pred_prob = np.zeros((x_test.shape[0], class_num))
        k = 0
        while k + batch_size < x_test.shape[0]:
            imgs = x_test[k:k + batch_size, :, :, :]
            pred_prob[k:k + batch_size] = predict_label(net, imgs)
            k += batch_size
        if k < x_test.shape[0]:
            net.blobs['data'].reshape(x_test.shape[0] - k, 3, 224, 224)
            imgs = x_test[k:, :, :, :]
            pred_prob[k:] = predict_label(net, imgs)

        yfull_test.append(pred_prob)

    pred_label = merge_several_prob(yfull_test, len(yfull_test))

    print ('{} files have been predicted.'.format(x_test.shape[0]))

    fr = open('files/labelmaptable.txt', 'r')
    rows = fr.readlines()
    fr.close()
    label_map_table = []
    for row in rows:
        row = row.strip('\n').split(' ')
        label_map_table.append(int(row[1]))

    with open('results/result.txt', 'w') as fw:
        for idx, label in enumerate(pred_label):
            fw.write(str(label_map_table[pred_label[idx]]) + '\t' + names[idx].split('.')[0] + '\n')

