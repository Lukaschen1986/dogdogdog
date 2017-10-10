# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: py_visualize.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年07月05日 星期三 10时51分29秒
#########################################################################

import os
import cv2
import numpy as np

fr = open('/home/banggui/caffe-root/caffe-master/examples/dogdogdog/files/trainval.txt', 'r')
rows = fr.readlines()
fr.close()

names = []
labels = []
for row in rows:
    row = row.strip('\n').split(' ')
    names.append(row[0])
    labels.append(int(row[1]))

each_class_sample = []
for label in range(100):
    samples = []
    for idx in range(len(names)):
        if label == labels[idx]:
            samples.append(idx)
    each_class_sample.append(samples)

for label in range(100):
    for name in each_class_sample[label][:5]:
        img = cv2.imread(os.path.join('/home/banggui/caffe-root/caffe-master/examples/dogdogdog/data/imgs/train/', names[name]))
        cv2.putText(img, str(labels[name]), (10, 100), cv2.FONT_HERSHEY_TRIPLEX, 4, (255, 0, 0))
        cv2.imshow('img', img)
        cv2.waitKey(0)

