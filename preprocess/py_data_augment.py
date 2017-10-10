# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: preprocess/py_data_augment.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年07月05日 星期三 23时01分26秒
#########################################################################

import numpy as np
import cv2
import os
import random

#获取训练集的数据
def get_imgs_list():
    fr = open('files/trainval.txt', 'r')
    rows = fr.readlines()
    fr.close()

    names = []
    labels = []
    for row in rows:
        row = row.strip('\n').split(' ')
        names.append(row[0])
        labels.append(int(row[1]))

    return names, labels

#将整个训练集按样本标签进行分类
def split_samples(names, labels):
    each_class_samples = []
    for label in range(100):
        samples = []
        for idx in range(len(names)):
            if labels[idx] == label:
                samples.append(idx)

        each_class_samples.append(samples)

    return each_class_samples

#从一个256*256的图像中随机裁剪出两个224x224的图像
def random_crop_img(img):
    new_imgs = []
    big_img = cv2.resize(img, (256, 256))
    x1 = random.randint(0, 40)
    y1 = random.randint(0, 40)
    new_imgs += [big_img[x1:x1 + 224, y1:y1 + 224]]

    x2 = random.randint(0, 40)
    y2 = random.randint(0, 40)
    new_imgs += [big_img[x2:x2 + 224, y2:y2 + 224]]

    return new_imgs

#旋转图像
def random_rotate_img(img):
    rows, cols, ch = img.shape
    new_imgs = []

    degree = random.randint(-45, 45)
    M = cv2.getRotationMatrix2D((rows / 2, cols / 2), degree, 1)
    new_imgs += [cv2.warpAffine(img, M, (rows, cols))]

    degree = random.randint(-45, 45)
    M = cv2.getRotationMatrix2D((rows / 2, cols / 2), degree, 1)
    new_imgs += [cv2.warpAffine(img, M, (rows, cols))]

    return new_imgs

#对图像进行预处理，每张经过预处理的图像可以得到9张新的图像
def preprocess(img):
    new_imgs = []

    flip_img = cv2.flip(img, 1)
    new_imgs += random_crop_img(img)
    new_imgs += random_rotate_img(img)
    new_imgs += random_crop_img(flip_img)
    new_imgs += random_rotate_img(flip_img)
    new_imgs += [flip_img]

    return new_imgs

def do_data_augment(names, labels, each_class_samples):
    imgs_dir = 'data/imgs/train'
    output_dir = 'data/imgs/data augment'
    count = 0

    #将新生成的图像重命名，写到new_trainval文件中
    with open('files/new_trainval.txt', 'w') as fw:
        for label in range(100):
            sample_num = len(each_class_samples[label])
            new_sample_num = sample_num
            for idx in each_class_samples[label]:
                new_imgs = []
                img = cv2.imread(os.path.join(imgs_dir, names[idx]))
                img = cv2.resize(img, (224, 224))
                #当每类样本的数量达到400时，就不再做数据扩展操作
                if new_sample_num < 400:
                    new_imgs = preprocess(img)
                    new_sample_num += 9
                new_imgs += [img]

                #保存新生成的图像
                for k in range(len(new_imgs)):
                    new_name = '{0:0>7}.jpg'.format(count)
                    cv2.imwrite(os.path.join(output_dir, new_name), new_imgs[k])
                    fw.write(new_name + ' ' + str(label) + '\n')
                    count += 1
            print 'label {} have {} new images.'.format(label, new_sample_num)

    print 'We generate total {} new images.'.format(count)

if __name__ == '__main__':
    names, labels = get_imgs_list()
    each_class_samples = split_samples(names, labels)
    do_data_augment(names, labels, each_class_samples)
