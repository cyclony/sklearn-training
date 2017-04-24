#梯度下降算法实现
import numpy as np

class Network:
    layers = None
    biases = None
    weights = None

    def __init__(self, sizes):
        self.layers = len(sizes)
        self.biases = [np.random.randn(m,1) for m in sizes[1:]]
        self.weights = [np.random.randn(m,n)     #第一层的输出（个数是第一层的神经元数量）是第二层的输入，第二层的输出（个数是第二层神经元数量）是第三场的输入
                        for n,m in zip(sizes[:-1], sizes[1:])]
    def sigmoid(self,z):
        return 1.0/(1.0 - np.exp(-z))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(w*a + b) #每一层神经元的输出，作为下一层神经元的输入
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None): #training_data((x,y)'s list)
        if test_data : n_test = len(test_data)
        n = len(training_data)
        mini_batches = [training_data[k:k+mini_batch_size]   #将training_data按照mini_batch的大小切成一块一块的list
                        for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            self.updata_mini_batch(mini_batch, eta)  #run gradient descent


