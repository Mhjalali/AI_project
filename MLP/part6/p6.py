from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

noise_range = 0.1
hidden_layer_sizes = (100, 100, 100)
max_iter = 3000
activation = 'relu'
solver = 'lbfgs'


def insert_noise(flatten):
    for j in range(len(flatten)):
        r = np.random.random()
        if r < noise_range:
            flatten[j] = np.random.randint(0, 256)
    return flatten


def generate_data(address: str):
    images = []
    noisy_images = []
    for img in os.listdir(address):
        s = os.path.join(address, img)
        if os.path.isfile(s):
            image = Image.open(s)
            array = np.array(image)
            flatten = array.flatten()
            flatten_c = flatten.copy()
            images.append(flatten)
            noisy_images.append(insert_noise(flatten_c))
    return images, noisy_images


images, noisy_images = generate_data("img")
train_x, test_x, train_y, test_y = train_test_split(noisy_images, images)
clf = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, solver=solver)
clf.fit(train_x, train_y)
p = clf.predict(test_x)
print(clf.score(test_y, p))

examples = []
for i in range(5):
    hold = random.randint(0, len(test_y))
    examples.append(hold)

for i in examples:
    t = test_y[i].reshape(16, 16)
    n = test_x[i].reshape(16, 16)
    NNP = p[i].reshape(16, 16)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(t)
    ax[1].imshow(n)
    ax[2].imshow(NNP)
    plt.show()
