import matplotlib.pyplot as pl
import numpy as np
from sklearn.neural_network import MLPRegressor
import math
from math import sin, cos

hidden_layer_sizes = (10, 10, 10)
max_iter = 2000
activation = 'tanh'
solver = 'lbfgs'
train_x, test_x, train_y, test_y = [], [], [], []

data_y = [1.1, 1.5, 2.5, 3, 3.5, 4, 4.5, 4.9, 4.3, 3.8, 3.6, 3.5, 3.8, 4, 5, 6, 6.2, 6.4, 7, 7.2, 6.8, 6, 5.1, 4.8, 4.5,
          4.8,
          5, 5.5, 6, 5.9, 5.7, 5, 4.8, 4.3, 4, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.1, 6.4, 5.8, 5, 4, 3.8, 3.3, 3]
data_x = []
for i in range(len(data_y)):
    data_x.append(i * 0.2)

for k in range(len(data_y)):
    if k % 3 == 0:
        test_x.append(k * 0.2)
        test_y.append(data_y[k])
    else:
        train_x.append(k * 0.2)
        train_y.append(data_y[k])

train_x = np.array(train_x).reshape(-1, 1)
train_y = np.array(train_y).reshape(-1, 1)
test_x = np.array(test_x).reshape(-1, 1)
clf = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, activation=activation,
                   solver=solver).fit(train_x,
                                      train_y)
p_y = clf.predict(test_x)

sum = 0
for i in range(len(p_y)):
    sum += ((test_y[i] - p_y[i]) ** 2) / (p_y[i] ** 2)
print(math.sqrt(sum))

# pl.plot(data_x, data_y, 'g')
pl.plot(test_x, test_y, 'r')
pl.plot(test_x, p_y, 'b')
pl.plot(train_x, train_y, 'black')

pl.legend(['reality', 'prediction', 'train'])
pl.show()
