import random
from sklearn import svm
import matplotlib.pyplot
import numpy as np


def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = matplotlib.pyplot.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


samples = []
label = []
func1 = '(x - 25)**2 + (y - 25)**2 - 225'
func2 = '(x + 25)**2 + (y + 25)**2 - 225'
func3 = '(x - 25)**2 + (y + 25)**2 - 225'
func4 = '(x + 25)**2 + (y - 25)**2 - 225'
n_train = 1000
for i in range(n_train):
    x = random.uniform(-50, 50)
    y = random.uniform(-50, 50)
    if x > 0 and y > 0:
        result = eval(func1)
    elif x > 0 and y < 0:
        result = eval(func3)
    elif x < 0 and y > 0:
        result = eval(func4)
    else:
        result = eval(func2)

    if result >= 0:
        label.append(1)
    else:
        label.append(-1)
    samples.append([x, y])

clf = svm.SVC(kernel='poly', degree=8, coef0=3)
clf.fit(samples, label)

test_n = 500
test_data = []
test_l = []
for i in range(test_n):
    x = random.uniform(-50, 50)
    y = random.uniform(-50, 50)
    if x > 0 and y > 0:
        result = eval(func1)
    elif x > 0 and y < 0:
        result = eval(func3)
    elif x < 0 and y > 0:
        result = eval(func4)
    else:
        result = eval(func2)
    if result >= 0:
        test_l.append(1)
        matplotlib.pyplot.scatter([x], [y], color="blue", marker="+", s=30, cmap='autumn')
    else:
        test_l.append(-1)
        matplotlib.pyplot.scatter([x], [y], color="red", marker="*", s=30, cmap='autumn')
    test_data.append([x, y])

correct = test_n
p = clf.predict(test_data)
for i in range(test_n):
    if p[i] != test_l[i]:
        correct -= 1  # :(
print("Accuracy: ", correct * 100 / test_n)

plot_svc_decision_function(clf)
matplotlib.pyplot.show()
