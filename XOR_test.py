import numpy as np
class XOR_Network:
    w1 = []
    b1 = 0
    w2 = []
    b2 = 0
    max_epochs = 0
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def fit_neuron(self, current_round, w,b, X, Y):
        target_w = w
        target_b = b
        for x,y in zip(X,Y):
            error = y - self.hardlim(np.dot(w, x)+b)
            w = w +error * x
            b = b + error
        current_round += 1
        print("weights is: ", w)
        print("bias is:", b)
        if (current_round == self.max_epochs):
            print('reached max peoch,precess ended!')
            return target_w, target_b
        elif (np.all(w == target_w)) and (target_b == b):
            print('weight is convergence on: ',target_w)
            return target_w, target_b
        else:
            return self.fit_neuron(current_round, w, b, X, Y)
    def hardlim(self, x):
        if x>0: return 1
        else: return 0

    def fit(self):
        X = np.array([[0,0],
             [0,1],
             [1,0],
             [1,1]])
        y = np.array([0,1,1,0])
        y1 = np.array([0,1,1,1])
        y2 = np.array([1,1,1,0])
        self.w1, self.b1 = self.fit_neuron(0, np.array([0,1]),0.5, X, y1)
        self.w2, self.b2 = self.fit_neuron(0, np.array([0,1]),0.5, X, y2)

    def precise(self, test_x, test_y):
        o1 = self.hardlim(np.dot(self.w1, test_x) + self.b1)
        o2 = self.hardlim(np.dot(self.w2, test_x) + self.b2)
        return o1 * o2  #模拟AND操作



test_x = [0.5, 0.5]
test_y = 1
net = XOR_Network(max_epochs=10)
net.fit()
print(net.precise(test_x,test_y))