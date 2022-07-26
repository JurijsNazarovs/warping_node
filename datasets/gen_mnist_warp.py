import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import sys

import importlib

data_gen_path = "/home/nazarovs/projects/data_generators"
data_path = './RotatingMnist'
sys.path.append(data_gen_path)
os.makedirs(data_path, exist_ok=True)

import rotating_mnist
importlib.reload(rotating_mnist)

from rotating_mnist import RotatingMnist
from matplotlib.pyplot import figure

digit = 3
data_gen = RotatingMnist(
    root='./',
    n_samples=1000,
    n_t=2,
    n_same_initial=1,
    initial_random_rotation=False,
    n_angles=1,
    min_angle=30,
    max_angle=30,
    frame_size=32,
    device='cpu',
    specific_digit=digit,
    # n_styles=10, #if ignored than all available np.infty
    mnist_data_path="%s/mnist-images-idx3-ubyte.gz" % data_gen_path,
    mnist_labels_path="%s/mnist-labels-idx1-ubyte.gz" % data_gen_path,
    name="ME_Rotating_MNIST")  # Generate data

data = data_gen.data.cpu().numpy()
source = data[:, 0]
target = data[:, -1]

plt.imshow(source[1, 0])
plt.savefig("./test1.png")
plt.imshow(target[1, 0])
plt.savefig("./test2.png")

data_gen.visualize(data[1],
                   plot_path='./test_data',
                   save_separate=True,
                   img_w=160,
                   img_h=160)

os.makedirs("%s/source_%d" % (data_path, digit), exist_ok=True)
os.makedirs("%s/target_%d" % (data_path, digit), exist_ok=True)
pickle.dump(source, open('%s/source_%d/source.pkl' % (data_path, digit), 'wb'))
pickle.dump(target, open('%s/target_%d/target.pkl' % (data_path, digit), 'wb'))
