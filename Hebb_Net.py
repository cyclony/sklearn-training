import numpy as np
class Hebb_Net:
    W = None
    def __init__(self, P, T):
        self.W = np.dot(T, P.T)
    def predict(self, data):
        return np.dot(self.W, data)

P = np.array([[1, -1, 1, -1],
             [1, 1, -1 , -1]]).T
T = np.array([[1, -1],
             [1, 1]]).T
data = np.array([1, -1, 1, -1])

h_net = Hebb_Net(P, T)
print(h_net.predict(data))