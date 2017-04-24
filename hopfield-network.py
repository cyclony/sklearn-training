#hopfield网络，有监督的学习网络，采用训练集数据对网络权值矩阵进行训练，满足分类要求;
import numpy as np

class Hopfield_network:#一个输出，判定1/0， 需要一个神经元
    init_weight = None
    init_b = 0
    r = 0
    def __init__(self, init_weight, init_b): #权值矩阵可以随意指定，只要线性可分，一定可以收敛
        self.init_weight = np.array(init_weight)
        self.init_b = init_b

    def train(self, data, targets): #输入三输入，一输出的一组训练数据，对神经网络的权值矩阵进行训练
        weight = self.init_weight
        b = self.init_b
        self.r += 1
        for sample, target in zip(np.array(data), np.array(targets)):
            e = target - self.hardlim(np.dot(weight , sample.reshape(-1,1))) #reshape(-1,1)相当于对矩阵求转置矩阵
            weight = weight + e * sample
            b = b + e
        if np.all(weight == self.init_weight): #权值矩阵经过训练没有变化，表示已经收敛
            print("final weights is: ")
            print(self.init_weight)
            print('round no is: '+ str(self.r))
            return
        else:                         #权值矩阵尚未收敛，持续递归训练（数学上可以证明只要线性可分，一定能收敛）
            self.init_weight = weight
            self.train(data, targets)



    def predict(self, tests):
        return self.hardlim(np.dot(self.init_weight, np.array(tests).reshape(-1,1)))#权值矩阵和输入矩阵转置求内积 w1p1+w2p2+...+WnPn

    def hardlim(self, x):
        if x >=0 : return 1
        else: return 0

init_weight = [0, -1]
data = [[1, 4],
        [1, 5],
        [2, 4],
        [2, 5],
        [3, 1],
        [3, 2],
        [4, 1],
        [4, 2]]
targets = [0,0,0,0,1,1,1,1]
test = [3, 2]


hfn = Hopfield_network(init_weight, 0)
hfn.train(data, targets)
print(hfn.predict(test))