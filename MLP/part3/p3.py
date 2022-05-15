import numpy as np
from sklearn.neural_network import MLPRegressor
import math
import random
import matplotlib.pyplot
from math import sin, cos

train_range = 10
test_range = 20
train_n = 1000
test_n = 1000
hidden_layer_sizes = (10, 10, 10)
max_iter = 2000
activation = 'relu'
solver = 'lbfgs'
func = "2*x + 5*y - z + 1"
train_x, train_y, train_z, test_x, test_y, test_z, train_f, test_f = [], [], [], [], [], [], [], []


def generate_data():
    for i in range(train_n):
        x = random.uniform(-train_range, train_range)
        y = random.uniform(-train_range, train_range)
        z = random.uniform(-train_range, train_range)
        f = eval(func)
        train_x.append(x)
        train_y.append(y)
        train_z.append(z)
        train_f.append(f)
    for i in range(test_n):
        x = random.uniform(-test_range, test_range)
        y = random.uniform(-test_range, test_range)
        z = random.uniform(-test_range, test_range)
        test_x.append(x)
        test_y.append(y)
        test_z.append(z)


generate_data()
train_x = np.array(train_x).reshape(-1, 1)
train_y = np.array(train_y).reshape(-1, 1)
train_z = np.array(train_z).reshape(-1, 1)
train_f = np.array(train_f).reshape(-1, 1)
test_x = np.array(test_x).reshape(-1, 1)
test_y = np.array(test_y).reshape(-1, 1)
test_z = np.array(test_z).reshape(-1, 1)

train = np.hstack((train_x, train_y, train_z))
clf = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, activation=activation,
                   solver=solver).fit(train,
                                      train_f)
test = np.hstack((test_x, test_y, test_z))
test_f = clf.predict(test)

real_f = []
for i in range(len(test_x)):
    x = test_x[i]
    y = test_y[i]
    z = test_z[i]
    real_f.append(eval(func))

fig = matplotlib.pyplot.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(test[:, 0], real_f, s=50, c='b', marker="+", label='reality')
ax1.scatter(test[:, 0], test_f, s=30, c='r', marker="*", label='prediction')

fig2 = matplotlib.pyplot.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(test[:, 1], real_f, s=50, c='b', marker="+", label='reality')
ax2.scatter(test[:, 1], test_f, s=30, c='r', marker="*", label='prediction')

fig3 = matplotlib.pyplot.figure()
ax3 = fig3.add_subplot(111)
ax3.scatter(test[:, 2], real_f, s=50, c='b', marker="+", label='reality')
ax3.scatter(test[:, 2], test_f, s=30, c='r', marker="*", label='prediction')

sum = 0
for i in range(len(test_f)):
    sum += ((real_f[i] - test_f[i]) ** 2) / (real_f[i] ** 2)
print(math.sqrt(sum))

matplotlib.pyplot.legend()
matplotlib.pyplot.show()
