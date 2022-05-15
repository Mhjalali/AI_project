import matplotlib.pyplot
import numpy as np
from sklearn.neural_network import MLPRegressor
import random
from math import sin, cos

train_range = 10
test_range = 20
train_n = 1000
test_n = 1000
hidden_layer_sizes = 100
max_iter = 2000
activation = 'tanh'
solver = 'lbfgs'
func = "5 * sin(5/x)"
train_x, test_x, train_y, test_y = [], [], [], []


def generate_data():
    for i in range(train_n):
        x = random.uniform(-train_range, train_range)
        y = eval(func)
        train_x.append(x)
        train_y.append(y)
    for i in range(test_n):
        x = random.uniform(-test_range, test_range)
        test_x.append(x)


generate_data()
train_x = np.array(train_x).reshape(-1, 1)
train_y = np.array(train_y).reshape(-1, 1)
test_x = np.array(test_x).reshape(-1, 1)
clf = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, activation=activation,
                   solver=solver).fit(train_x,
                                      train_y)
test_y = clf.predict(test_x)

fig = matplotlib.pyplot.figure()
ax1 = fig.add_subplot(111)

real_y = []
for i in test_x:
    x = i
    real_y.append(eval(func))

ax1.scatter(test_x, real_y, s=50, c='b', marker="+", label='reality')
ax1.scatter(test_x, test_y, s=50, c='r', marker="*", label='prediction')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()
