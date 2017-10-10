# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: py_visualize_conv.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年07月08日 星期六 10时46分49秒
#########################################################################

from matplotlib.pyplot import *
import matplotlib
import numpy as np
import caffe
import cv2
import os

def read_data(transformer):
    path = 'data/imgs/train/587026,3367516892.jpg'
    img = caffe.io.load_image(path)
    trans_img = transformer.preprocess('data', img)

    return trans_img

def get_features(net, img):
    net.blobs['data'].data[...] = img
    net.forward()
    conv5_3 = net.blobs['conv5_3'].data

    return conv5_3

caffe.set_mode_gpu()
caffe.set_device(0)
mu = np.array([104, 117, 123])
model_deploy = 'prototxt/VGG16_deploy.prototxt'
model_weights = 'models/Baidu_VGG16_1_iter_30000.caffemodel'
net = caffe.Net(model_deploy, model_weights, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))
net.blobs['data'].reshape(1, 3, 224, 224)

img = read_data(transformer)
feats = get_features(net, img)
feats = feats[0]
print feats.shape
feats = feats.sum(axis = 0)
feats = feats.reshape(1, 14 * 14)
feats = feats / (max(feats) - min(feats))
feats = np.vstack((feats, feats, feats))
feats = feats.reshape(feats.shape[1], feats.shape[0])
print feats
print feats.shape
cdict = {'red': feats.tolist(), 'green': feats.tolist(), 'blue':feats.tolist()}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
pcolor(np.random.rand(14, 14), cmap = my_cmap)
colorbar()
show()
