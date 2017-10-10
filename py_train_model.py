# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: py_train_model.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年07月04日 星期二 10时59分39秒
#########################################################################

import os
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

weights = 'ResNet-101-model.caffemodel'
for idx in range(3,4):
    solver_path = 'prototxt/ResNet101_solver{}.prototxt'.format(idx)
    solver = caffe.SGDSolver(solver_path)
    solver.net.copy_from(weights)
    solver.solve()
