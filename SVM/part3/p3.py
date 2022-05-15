import os
import numpy as np
from PIL import Image
from sklearn import svm

lables = ['S', 'W', '2', '3', '7']


def data_maker(train_set, train_labels, test_set, test_labels):
    for l in lables:
        folder = os.path.join("data", l)
        i = 0
        for img in os.listdir(folder):
            img = os.path.join(folder, img)
            if os.path.isfile(img):
                image = Image.open(img)
                array = np.array(image)
                flatten = array.flatten()
                if i % 100 == 0:
                    train_set.append(flatten)
                    train_labels.append(l)
                else:
                    test_set.append(flatten)
                    test_labels.append(l)
            i += 1


train_set, train_labels, test_set, test_labels = [], [], [], []
data_maker(train_set, train_labels, test_set, test_labels)
clf = svm.SVC(kernel='poly', degree=5, coef0=3)
clf.fit(train_set, train_labels)
p = clf.predict(test_set)
correct = 0
for i in range(len(test_set)):
    if p[i] == test_labels[i]:
        correct += 1
print(correct * 100 / len(test_set))
