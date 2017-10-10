# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: py_random_sample.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年07月06日 星期四 14时22分37秒
#########################################################################

import os
import cv2
import numpy as np

fr = open('files/trainval.txt', 'r')
rows = fr.readlines()
fr.close()

x_train = []
y_train = []
for row in rows:
    row = row.strip('\n').split(' ')
    x_train.append(row[0])
    y_train.append(int(row[1]))

each_class_sample = []
for label in range(100):
    samples = []
    for idx in range(len(x_train)):
        if label == y_train[idx]:
            samples.append(idx)

    each_class_sample.append(samples)

for label in range(100):
    if not os.path.exists('data/imgs/samples/' + str(label)):
        os.mkdir('data/imgs/samples/' + str(label))

    count = 1
    for idx in each_class_sample[label]:
        if y_train[idx] == label:
            img = cv2.imread('data/imgs/train/' + x_train[idx])
            cv2.imwrite('data/imgs/samples/' + str(label) + '/' + x_train[idx], img)

            if count == 10:
                break
            count += 1

