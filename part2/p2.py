import os
import numpy as np
from PIL import Image
from sklearn import svm


def data_train(address: str):
    images = []
    labels = []
    for img in os.listdir(address):
        s = os.path.join(address, img)
        if os.path.isfile(s):
            image = Image.open(s)
            array = np.array(image)
            flatten = array.flatten()
            images.append(flatten)
            labels.append(int(img[0]))
    return images, labels


def data_test(address: str):
    images = []
    labels = []
    for img in os.listdir(address):
        s = os.path.join(address, img)
        if os.path.isfile(s):
            image = Image.open(s)
            array = np.array(image)
            flatten = array.flatten()
            images.append(flatten)
            labels.append(int(img[0]))
    return images, labels


train_set, train_labels = data_train("train")
test_set, test_labels = data_test("test")
clf = svm.SVC(kernel='poly', degree=5, coef0=3)
clf.fit(train_set, train_labels)
p = clf.predict(test_set)

correct = 0
for i in range(len(test_set)):
    if p[i] == test_labels[i]:
        correct += 1
print(correct * 100 / len(test_set))
