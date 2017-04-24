# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import json

#有监督的仿逆规则hebb网络,网络输出T = WP, 要使得目标结果和实际输出结果尽可能接近，则必须使得WP -T最小，假设为零，则W=TP的逆矩阵，最小
class F_Inverse_Net:
    W = None
    def __init__(self, T, P): #通过训练数据集（输入和目标输出）构建权值矩阵
        inversed_P = np.linalg.pinv(P)        #利用numpy的仿逆函数求得P的仿逆矩阵
        self.W = np.dot(T, inversed_P)
    
    def predict(self, data):
        return np.dot(self.W, data)

P = np.array([[1,-1,1, -1],
     [1,1,-1,-1]]).T
T = np.array([[1,-1],[1,1]]).T
data = np.array([1,1,-1,-1]).T
  
fi_net = F_Inverse_Net(T, P)
print(fi_net.predict(data))
