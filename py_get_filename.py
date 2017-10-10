# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: py_get_filename.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年07月03日 星期一 12时42分59秒
#########################################################################

import os

fr = open('url/train_url.txt', 'r')
rows = fr.readlines()
fr.close()

with open('train.txt', 'w') as fw:
    for row in rows:
        row = row.strip('\n').split()
        fw.write(row[0] + '.jpg ' + row[1] + '\n')

