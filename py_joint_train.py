# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: py_joint_train.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年07月14日 星期五 02时05分52秒
#########################################################################

import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.utils import np_utils

def load_train_data(filename):
    path = os.path.join('feats', filename)
    feats = np.load(path)
    y_train = feats[:, -1]
    x_train = feats[:, 0:feats.shape[1] - 1]
    y_train = np_utils.to_categorical(y_train, 100)

    return x_train, y_train

def load_test_data(filename):
    path = os.path.join('feats', filename)
    feats = np.load(path)
    names = feats[:, -1]
    x_train = feats[:, 0:feats.shape[1] - 1]
    x_train = x_train.astype(float)

    return x_train, names

def get_model():
    model = Sequential()
    model.add(Dense(4096, input_dim = 2048 * 3, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation = 'softmax'))

    sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer = sgd, loss = 'categotical_crossentroy')

    return model

def save_model(model, modelname):
    json_str = model.to_json()
    json_name = 'architecture_' + modelname + '.json'
    weight_name = 'weights_' + modelname + '.h5'
    open(os.path.join('models/joint_models/', json_name), 'w').write(json_str)
    model.save_weights(os.path.join('models/joint_models/', weight_name), overwrite = True)

def read_model(modelname):
    json_name = 'architecture_' + modelname + '.json'
    weight_name = 'weights_' + modelname + '.h5'
    model = model_from_json(open(os.path.join('models/joint_models', json_name)).read())
    model.load_weights(os.path.join('model/joint_models', weight_name))
    return model

def run_cross_validation(folds = 5, nb_epoch = 10, split = 0.2, modelstr = ''):
    batch_size = 128
    for idx in range(folds):
        x_train, y_train = load_train_data('train' + str(idx) + '.npy')
        x_val, y_val = load_train_data('val' + str(idx) + '.npy')
        model = get_model()
        model.fit(x_train, y_train, batch_size = batch_size, nb_epoch = nb_epoch,
                  show_accuracy = True, verbose = 1, validation_data = (x_val, y_val))
        save_model(model, 'joint_model_' + str(idx))


