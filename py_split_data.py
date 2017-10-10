# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: py_split_data.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年07月03日 星期一 14时06分13秒
#########################################################################

import numpy as np
import random
import matplotlib.pyplot as plt

fr = open('files/train.txt', 'r')
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
#
each_class_num = []
each_class_sample = []
for label in range(100):
    count = 0
    samples = []
    for idx in range(size):
        if label == y_train[idx]:
            count += 1
            samples.append(idx)

    each_class_num.append(count)
    each_class_sample.append(samples)

x_test_idx = []
x_train_idx = []
for label in range(100):
    num = len(each_class_sample[label])
    test_num = int(num * 0.1)
    for idx in each_class_sample[label][:test_num]:
        x_test_idx.append(idx)
    for idx in each_class_sample[label][test_num:]:
        x_train_idx.append(idx)

with open('files/trainval.txt', 'w') as fw:
    for idx in x_train_idx:
        fw.write(x_train[idx] + ' ' + str(y_train[idx]) + '\n')

with open('files/test.txt', 'w') as fw:
    for idx in x_test_idx:
        fw.write(x_train[idx] + ' ' + str(y_train[idx]) + '\n')

#fw = open('trainval/train.txt', 'w')
#for idx in range(0, 15000):
#    fw.write('data/imgs/train/' + x_train[index[idx]] + ' ' + str(y_train[index[idx]]) + '\n')
#fw.close()
#
#fw = open('trainval/val.txt', 'w')
#for idx in range(15000, size):
#    fw.write('data/imgs/train/' + x_train[index[idx]] + ' ' + str(y_train[index[idx]]) + '\n')
#fw.close()
#
labels = np.arange(100)
label_num = np.zeros(100)
label = set()
for idx in range(size):
    label_num[int(y_train[idx])] += 1

count = 0
for idx in range(100):
    if label_num[idx] == 0:
        print idx
        count +=1

print count
print min(label_num)
print max(label_num)
print label_num
