import numpy as np
from functools import reduce
from operator import mul

def ss_hardlims(weights, inputs, b):
    w_matrix = np.matrix(weights)
    i_matrix = np.matrix(inputs).T
    result = w_matrix * i_matrix + b
    if result>= 0: return 1
    if result <0: return -1

weights = [0,1,0]
inputs = [1,-1,-1]
print(ss_hardlims(weights, inputs, 0))

#Hamming 网络:有监督的分类网络，求解二值识别模式的问题；
#将典型样本训练数据作为前馈网络的权值矩阵
class Hamming:
    purelin_weights_matrix = None
    poslin_weights_matrix = np.matrix([[1, -0.5],   #
                                       [-0.5,1]])
    target = None

    b=[[3],#保证前馈层输出不为负数
       [3]]

    def __init__(self, purelin_weights, target):# 前馈层的权重网络,使用两个标准向量初始化网络，通过和两个标准向量进行对比；和目标结果向量
        self.purelin_weights_matrix = np.matrix(purelin_weights)
        self.target = target

    def purelin(self, inputs): #前馈层设计,放大输入向量
        i_matrix = np.matrix(inputs).T
        return self.purelin_weights_matrix * i_matrix + self.b

    def poslin(self, inputs):#反馈网络，放大大数，缩小小数，负数调整为0，最终收敛
        results = self.poslin_weights_matrix * inputs
        results[results<0] = 0 #如果结果向量中存在负值，则将负值转化为0；其他不变  [results<0]是数组的布尔值作为索引
        if np.all(results == inputs): return results #如果输入向量和调整后的结果向量相等，则说明已经收敛，返回结果，否则继续迭代。
        else: #如果输入向量和调整后的结果向量不相等，说明尚未收敛
            return self.poslin(results)  #将调整后结果作为新的输入进行迭代

    def predict(self, inputs):
        predicted_target =  self.poslin(self.purelin(inputs))
        if np.all(self.target * predicted_target) == 0:
            return np.negative(self.target)
        else:
            return self.target

inputs = [-1,-1,-1]
sample_data = [[1, -1, -1],   #这两个输入分别代表两类典型数据
               [1, 1, -1]]
hamming_network = Hamming(sample_data, [1 , 0])
print(hamming_network.predict(inputs))


