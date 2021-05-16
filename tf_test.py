#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"

#from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np



import xlrd
import xlwt

read=xlrd.open_workbook('weather.xls')
data=read.sheets()[0]
print(data.nrows)
print(data.ncols)

t1 = data.col_values(11)[1:]    # "Humidity9am"
t1 = np.array(t1).astype(np.float)  # list array  to numpy
len=t1.shape[0]
X=np.reshape(t1, (len,1))
X=np.append(X,np.reshape(np.array(data.col_values(12)[1:]).astype(np.float) ,  (len,1)), axis=1)   # Humidity3pm
X=np.append(X,np.reshape(np.array(data.col_values(4)[1:]).astype(np.float),  (len,1)), axis=1)     # Sunshine

t1 = data.col_values(20)[1:]    # "Label"
Y = np.array( t1).astype(np.int)  # list array  to numpy