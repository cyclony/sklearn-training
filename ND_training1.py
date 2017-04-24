import numpy as np
import sklearn.datasets
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class GD_Network:
    weights = []
    training_data = []
    alpha = 0
    epochs = 0
    X = []
    Y = []
    costs = []

    def __init__(self, inputs, outputs, alpha=0.0001, epochs=500):
        self.training_data = list(zip(inputs, outputs))
        np.random.shuffle(self.training_data)
        self.alpha = alpha
        self.epochs = epochs
        self.X = inputs
        self.Y = outputs

    def cost_function(self):
        n = len(self.training_data)
        cost = 0
        for X, Y in self.training_data:
            cost += (np.dot(self.weights, X) - Y) **2
        return cost/n

    def fit(self, init_w):
        self.weights = init_w
        for _ in range(self.epochs):
            print("current weight is:",self.weights)
            outputs = self.net_input(self.X)
            errors = self.Y - outputs
            self.weights = self.weights + self.alpha * self.X.T.dot(errors)
            self.costs.append((errors**2).sum())
            print("error is : ", (errors**2).sum())
        return self

    def train_weights(self, w):
        self.weights = w
        for _ in range(self.epochs):
            print("current epoch cost is:", self.cost_function())
            for X, Y in self.training_data:
                w = w - self.alpha * self.gradient(w, X, Y)
            self.weights = w
        print("final weight's is: ",self.weights )

    def net_input(self, X):
        return np.dot(X, self.weights)

    def gradient(self, w, X, Y):
        return (np.dot(w, X) - Y)*X

    def precise(self, test_data):
        return np.dot(self.weights, test_data)


#iris = sklearn.datasets.load_iris()
iris = sklearn.datasets.load_iris()

sc = StandardScaler()


x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

"""
for i in range(4):
    x_train_std[:,i] = (x_train[:,i] - x_train[:,i].mean())/x_train[:,i].std()
    x_test_std[:,i] = (x_test[:,i] - x_test[:,i].mean())/x_test[:,i].std()
"""

net = GD_Network(x_train_std, y_train, alpha=0.0001, epochs=200)
#net.train_weights(iris.data[random.randint(0, len(iris.data)-1)])
net.fit(x_train_std[random.randint(0, len(y_train)-1)])
test_data = list(zip(x_test_std, y_test))
index  = random.randint(0, len(x_test))
infact_output = test_data[index][1]
infact_input = test_data[index][0]
precise_result = net.precise(infact_input)
print("infact input is",str(infact_input))
print("infact output is", str(infact_output))
print("precised output is", str(precise_result))
plt.plot(range(1, len(net.costs) + 1),np.log10(net.costs), marker='o')
plt.xlabel('Epochs')
plt.ylabel('errors')
plt.title('Adaline - Learning rate 0.0001')
plt.show()
"""
inputs = np.array(iris.data)
inputs_std = np.copy(inputs)
inputs_std[:,0] = (inputs[:,0] - inputs[:,0].mean())/inputs[:,0].std()
inputs_std[:,1] = (inputs[:,1] - inputs[:,1].mean())/inputs[:,1].std()
inputs_std[:,2] = (inputs[:,2] - inputs[:,2].mean())/inputs[:,2].std()
inputs_std[:,3] = (inputs[:,3] - inputs[:,3].mean())/inputs[:,3].std()


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
net1 = GD_Network(inputs_std, iris.target, alpha=0.0001, epochs=100).fit(inputs_std[random.randint(0, len(iris.data)-1)])
ax[0].plot(range(1, len(net1.costs) + 1),np.log10(net1.costs), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Sum-squared-error')
ax[0].set_title('Adaline - Learning rate 0.01')

net2 = GD_Network(iris.data, iris.target, alpha=0.0001, epochs=10).fit(iris.data[random.randint(0, len(iris.data)-1)])
ax[1].plot(range(1, len(net2.costs) + 1),net2.costs, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()
"""
