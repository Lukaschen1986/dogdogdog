# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: py_kflods.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年07月04日 星期二 10时22分30秒
#########################################################################

import numpy as np
import os
import random

from sklearn.cross_validation import KFold

fr = open('files/trainval.txt', 'r')
rows = fr.readlines()
fr.close()

x_train = []
y_train = []

for row in rows:
    row = row.strip('\n').split(' ')
    x_train.append(row[0])
    y_train.append(int(row[1]))

size = len(x_train)
index = np.arange(size)
random.shuffle(index)
#labels = set(y_train)
#label134tolabel100 = dict([label, i] for i, label in enumerate(labels))
#label100tolabel134 = dict([i, label] for i, label in enumerate(labels))
#
#with open('files/labelmaptable.txt', 'w') as fw:
#    for idx in label100tolabel134.keys():
#        fw.write(str(idx) + ' ' + str(label100tolabel134[idx]) + '\n')
#
#for idx in range(size):
#    y_train[idx] = label134tolabel100[y_train[idx]]

#将整个训练集部分，划分成5组，每次4组做训练，1组做验证
folds = 5
random_state = 20
num_fold = 0
#调用KFold函数，对数据进行随机划分
kf = KFold(len(x_train), n_folds = folds, shuffle = True, random_state = random_state)
for train, val in kf:
    with open('trainval/train_{}.txt'.format(num_fold), 'w') as fw:
        for idx in range(len(train)):
            fw.write('data/imgs/train/' + x_train[train[idx]] + ' ' + str(y_train[train[idx]]) + '\n')

    with open('trainval/val_{}.txt'.format(num_fold), 'w') as fw:
        for idx in range(len(val)):
            fw.write('data/imgs/train/' + x_train[val[idx]] + ' ' + str(y_train[val[idx]]) + '\n')
    num_fold += 1
