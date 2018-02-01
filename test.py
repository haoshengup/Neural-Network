import numpy as np
from sklearn import datasets
from sklearn import preprocessing
import NetWork_Adam as NetWork

iris = datasets.load_iris()
data = iris.data.T
target = iris.target[:, np.newaxis].T

t = np.ones((3, 150))
for j in range(target.shape[1]):
    if target[0, j] == 0:
        t[0, j] = 1
        t[1, j] = 0
        t[2, j] = 0
    elif target[0, j] == 1:
        t[0, j] = 0
        t[1, j] = 1
        t[2, j] = 0
    elif target[0, j] == 2:
        t[0, j] = 0
        t[1, j] = 0
        t[2, j] = 1

net = NetWork.NetWork([4,16,3])
train_data = data[:,0:140]
train_t = t[:,0:140]
test_data = data[:,140:150]
test_t = t[:,140:150]
training_data, scaler = net.normalization(train_data)
testing_data = net.normalization(test_data, scaler)[0]
e_mv_out = net.Adam(training_data = training_data,target = t,eta = [0.001,0.9,0.999], epoches=500,mini_batch_size=70)
net.evaluate(training_data,train_t, e_mv_out)
net.evaluate(testing_data, test_t, e_mv_out)