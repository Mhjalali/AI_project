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
func = '5*x + 4*y - 10'
n_train = 1000
for i in range(n_train):
    x = random.uniform(-500, 500)
    y = random.uniform(-500, 500)
    result = eval(func)
    if result >= 0:
        label.append(1)
    else:
        label.append(-1)
    samples.append([x, y])

clf = svm.SVC(kernel='linear')
clf.fit(samples, label)

test_n = 500
test_data = []
test_l = []
for i in range(test_n):
    x = random.uniform(-500, 500)
    y = random.uniform(-500, 500)
    result = eval(func)
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
