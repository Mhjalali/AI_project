from sklearn.neural_network import MLPClassifier
import os
import numpy as np
from PIL import Image

hidden_layer_sizes = (1500)
max_iter = 1000
activation = 'relu'
solver = 'adam'


def generate_data(address: str) -> tuple:
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


train_set, train_labels = generate_data('train')
test_set, test_labels = generate_data('test')
clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, solver=solver)
clf.fit(train_set, train_labels)
p = clf.predict(test_set)

correct = 0
for i in range(len(test_set)):
    if p[i] == test_labels[i]:
        correct += 1
print(correct * 100 / len(test_set))
