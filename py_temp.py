# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: py_visualize_conv.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年07月08日 星期六 10时44分22秒
#########################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

y = np.random.randint(1,100,40)
y = y.reshape((5,8))
df = pd.DataFrame(y,columns=[x for x in 'abcdefgh'])
sns.heatmap(df,annot=True)
plt.show()


