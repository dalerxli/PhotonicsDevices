print(__doc__)
import sys
import os
import numpy as np

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

def cur_file_dir():
    # 获取脚本路径
    path = sys.path[0]

    if os.path.isdir(path):
        return path

    elif os.path.isfile(path):
        return os.path.dirname(path)

path = cur_file_dir()

# The digits dataset
digits = datasets.load_digits()
images = digits.images
target = digits.target
shape1 = images.shape[0]
for i in range(0, shape1):
    im = (images[i])
    # 生成量化qua数据
    # im = (images[i]*6.25//20)*20
    # 生成0~1数据
    for j in range(0, 8):
        for k in range(0, 8):
            if im[j, k] > 0:
                im[j, k] = 100

    # 生成qua数据
    # np.savetxt(path + '\\\simulation\\MNIST\\mnist_sklearn_qua\\ma_' + str(i) + '.txt', im, fmt="%d",
    #            delimiter=" ")

    # 生成0~1数据
    np.savetxt(path + '\\\simulation\\MNIST\\mnist_sklearn_dig\\ma_' + str(i) + '.txt', im, fmt="%d",
               delimiter=" ")

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
# _, axes = plt.subplots(2, 4)
# images_and_labels = list(zip(digits.images, digits.target))
# for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
#     ax.set_title('Training: %i' % label)
#
#
# plt.show()