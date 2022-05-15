import matplotlib.pyplot
import numpy as np
from sklearn.neural_network import MLPRegressor
import random
from math import sin, cos

train_range = 5
test_range = 10
train_n = 2000
test_n = 2000
hidden_layer_sizes = (10, 10, 10)
max_iter = 2000
activation = 'tanh'
solver = 'lbfgs'
noise_range = 0.9
const = 0.9
func = "sin(5 * x)"
train_x, test_x, train_y, test_y = [], [], [], []
noise_type = input("Enter noise type: ")


def generate_data():
    for i in range(train_n):
        x = random.uniform(-train_range, train_range)
        y = eval(func)
        if noise_type == 'const':
            noise = random.uniform(-const, const)
        else:
            noise = random.uniform(-noise_range * y, noise_range * y)
        train_x.append(x)
        train_y.append(y + noise)
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
ax1 = fig.add_subplot(1, 1, 1)

real_y = []
for i in test_x:
    x = i
    real_y.append(eval(func))

ax1.scatter(test_x, real_y, s=20, c='b', marker="+", label='reality')
ax1.scatter(test_x, test_y, s=20, c='r', marker="*", label='prediction')
ax1.scatter(train_x, train_y, s=10, c='black', marker=".", label='train')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()
