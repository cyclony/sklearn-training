import neutronNetworkDeepLearning.Network as Network
import neutronNetworkDeepLearning.mnist_loader as mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network.Network([784, 10, 10])
net.SGD(training_data, 30, 10, 3, test_data=test_data)