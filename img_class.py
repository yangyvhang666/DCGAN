#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 将一个文件夹下图片按比例分在两个文件夹下，比例改0.7这个值即可
import os
import random
import shutil
from shutil import copy2
trainfiles = os.listdir('D:/data/face/img_align_celeba/')
num_train = len(trainfiles)
print( "num_train: " + str(num_train) )
index_list = list(range(num_train))
print(index_list)
random.shuffle(index_list)
num = 0
train_1 = 'D:/data/face/train_1'
train_2 = 'D:/data/face/train_2'
train_3 = 'D:/data/face/train_3'
train_4 = 'D:/data/face/train_4'
for i in index_list:
    fileName = os.path.join('D:/data/face/img_align_celeba/', trainfiles[i])
    if num < num_train*0.25:
        print(str(fileName))
        copy2(fileName, train_1)
    elif num > num_train*0.25 and num < num_train*0.5:
        copy2(fileName, train_2)
    elif num > num_train*0.5 and num < num_train * 0.75:
        copy2(fileName,train_3)
    else:
        copy2(fileName,train_4)
    num += 1