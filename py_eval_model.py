# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: py_eval_model.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年07月04日 星期二 23时06分16秒
#########################################################################

import os
import numpy as np
import caffe
import cv2

#导入测试集图像
def load_test_data(transformer):
    fr = open('files/test.txt', 'r')
    rows = fr.readlines()
    fr.close()

    names = []
    x_test = []
    y_test = []
    count = 0
    for row in rows:
        row = row.strip('\n').split(' ')
        #读入测试集图像
        img = caffe.io.load_image(os.path.join('data/imgs/train', row[0]))
        #对每张图像先进行预处理
        transformered_imgs = transformer.preprocess('data', img)
        x_test.append(transformered_imgs.copy())
        y_test.append(int(row[1]))
        names.append(row[0])
        count += 1

        if count % 1000 ==  0:
            print ('load {}/{} test files.'.format(count, len(rows)))

    print ('load {}/{} test files'.format(count, len(rows)))
    return x_test, y_test, names

#得到预测图像属于每个类别的概率值
def predict_label(net, imgs):
    #将图像输入到网络中
    net.blobs['data'].data[...] = imgs
    #网络进行前向计算，得到输出值
    output = net.forward()
    #获取概率输出
    output_prob = output['prob']

    return output_prob

#计算多个概率模型的概率平均值
def merge_several_prob(probs, folds):
    a = probs[0]
    a = np.array(a)
    for idx in range(1, folds):
        a += np.array(probs[idx])

    a /= folds
    return a.argmax(axis = 1)

#显示预测错误的图像
def show_error_img(names, pred_label, y_test):
    #for idx in range(len(pred_label)):
    #    if pred_label[idx] != y_test[idx]:
    #        img = cv2.imread(os.path.join('data/imgs/train/', names[idx]))
    #        img = cv2.resize(img, (256, 256))
    #        print 'real:{}, pred:{}'.format(y_test[idx], pred_label[idx])
    #        cv2.putText(img, str(y_test[idx]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
    #        cv2.putText(img, str(pred_label[idx]), (10, 100), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255))
    #        cv2.imshow('img', img)
    #        cv2.waitKey(0)


    with open('files/error_list.txt', 'w') as fw:
        for label in range(100):
            errors = []
            for idx in range(len(pred_label)):
                if label == y_test[idx] and pred_label[idx] != y_test[idx]:
                    errors.append(pred_label[idx])
                    fw.write(names[idx] + '\n')

if __name__ == '__main__':

    #设置模型测试时使用gpu模式
    caffe.set_mode_gpu()
    caffe.set_device(0)
    #图像要减的均值
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
    url = []
    for idx in range(folds):

        #网络的结构文件
        model_deploy = 'prototxt/' + net_model + '_deploy.prototxt'
        #网络的权值文件
        model_weights = 'models/Baidu_' + net_model + '_' + str(idx) + '_iter_10000.caffemodel'
        #导入网络模型
        net = caffe.Net(model_deploy, model_weights, caffe.TEST)
        #获取网络对输入数据的变换形式
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', mu)
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
        net.blobs['data'].reshape(batch_size, 3, 224, 224)

        #在第一个模型的时候导入测试数据
        if idx == 0:
            x_test, y_test, names = load_test_data(transformer)
            x_test = np.array(x_test)

        pred_prob = np.zeros((x_test.shape[0], class_num))
        k = 0
        #对图像进行预测
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

    #计算多个模型的概率平均值
    pred_label = merge_several_prob(yfull_test, len(yfull_test))

    print ('{} files have been predicted.'.format(x_test.shape[0]))

    count = 0
    for idx in range(len(pred_label)):
        if pred_label[idx] == y_test[idx]:
            count += 1

    #计算得到模型在测试数据集上的准确率
    print 'Accuracy rate is: {}'.format(1.0 * count / len(pred_label))

    #print pred_list
    #with open('pred_list.txt', 'w') as fw:
    #    for idx in range(len(pred_label)):
    #        fw.write(names[idx] + ' ' + str(y_test[idx]) + ' ' + str(pred_list[0][idx]) + ' ' + str(pred_list[1][idx]) + ' '
    #                 + str(pred_list[2][idx]) + ' ' + str(pred_list[3][idx]) + ' ' + str(pred_list[4][idx]) + '\n')
    #show_error_img(names, pred_label, y_test)
