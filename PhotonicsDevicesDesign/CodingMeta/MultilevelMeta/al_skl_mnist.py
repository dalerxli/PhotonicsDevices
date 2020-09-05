import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, metrics
from pandas import Series, DataFrame
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def cur_file_dir():
    # 获取脚本路径
    path = sys.path[0]

    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)

def plot_skl_mnist_result(num):
    path = cur_file_dir()

    digits = datasets.load_digits()
    images = digits.images
    target = digits.target
    #shape1 = images.shape[0]
    index_array = np.where(target == num)[0]
    # dig
    tr_dir = path + '\\data\\MNIST\\al_skl_minst_te_dig_tr\\'
    # qua
    # tr_dir = path + '\\data\\MNIST\\al_skl_minst_te_qua_tr\\'

    tr_file_list = os.listdir(tr_dir)

    for index_file, file in enumerate(tr_file_list):
        with open(tr_dir + file) as read_file:
            num_index = int(file.split('_')[1].split('.')[0])
            if index_array.__contains__(num_index):
                print(num_index)
                content = read_file.read()
                r_list = content.split('\t')
                r_list_float = []
                for index, item in enumerate(r_list):
                    r_list_float.append(float(item))
                # r_numpy = np.array(r_list_float) * (-1)
                # r_numpy = np.array(r_list_float)
                print('max', max(r_list_float))
                print('min', min(r_list_float))

                r_numpy = (np.array(r_list_float) * (-1))
                #r_numpy = (np.array(r_list_float)*(-1))[0:700, ]
                x = np.linspace(500, 2000, 1500)
                #x = np.linspace(500, 1200, 700)
                plt.legend()
                plt.plot(x, r_numpy)

    plt.ylim(0, 1)
    plt.show()


# sklearn mnist对应的结果、均值计算及训练
def plot_skl_mnist_result_mean(num_list):
    path = cur_file_dir()

    digits = datasets.load_digits()
    images = digits.images
    target = digits.target
    shape1 = images.shape[0]

    res = DataFrame()

    r_list_all = []

    r_numpy_div = np.zeros((10, 10))

    for num in num_list:

        num_sum = 0
        r_numpy_sum = np.zeros((700,))

        index_array = np.where(target == num)[0]

        tr_dir = path + '\\data\\MNIST\\al_skl_minst_te_dig_tr\\'
        tr_file_list = os.listdir(tr_dir)

        for index_file, file in enumerate(tr_file_list):
            with open(tr_dir + file) as read_file:
                num_index = int(file.split('_')[1].split('.')[0])
                if index_array.__contains__(num_index):
                    num_sum = num_sum + 1
                    content = read_file.read()
                    r_list = content.split('\t')
                    r_list_float = []
                    for index, item in enumerate(r_list):
                        r_list_float.append(float(item))
                    # r_numpy = np.array(r_list_float) * (-1)
                    # r_numpy = np.array(r_list_float)
                    r_numpy = (np.array(r_list_float)* (-1))[0:700, ]
                    # 采样点
                    # r_numpy_sample = r_numpy[[0, 50, 100, 200, 300, 350, 400, 450, 500, 550, 600]]
                    #r_numpy_sample_insert = np.insert(r_numpy_sample, 0, [num], axis=0)
                    #r_numpy_sample_insert = np.append(r_numpy_sample, [num], axis=0)
                    r_numpy_sample_insert = np.append(r_numpy, [num], axis=0)
                    # 插入采样点
                    # r_numpy_series = Series(r_numpy_sample_insert)
                    # 插入全部数据
                    r_numpy_series = Series(r_numpy_sample_insert)
                    res = res.append(r_numpy_series, ignore_index=True)
                    r_numpy_sum += r_numpy

        x = np.linspace(500, 1200, 700)
        plt.legend()
        print(num_sum)
        plt.plot(x, r_numpy_sum/(num_sum), label=str(num))
        r_numpy_sum_mean = r_numpy_sum / (num_sum)
        r_list_all.append(r_numpy_sum_mean)
    plt.ylim(0, 1)
    plt.show()

    # for i in [0, 1, 6, 8, 2, 3, 4, 5, 7, 9]:
    #     for j in [0, 1, 6, 8, 2, 3, 4, 5, 7, 9]:
    #         r_numpy_div[i, j] = np.linalg.norm(r_list_all[i] - r_list_all[j])

    # r_numpy_div_pd = DataFrame(r_numpy_div)
    # r_numpy_div_pd.to_excel('all-data.xlsx')

    res = shuffle(res)

    data = res.iloc[:, 0:res.shape[1]-1].values
    target = res.iloc[:, res.shape[1]-1].values

    # Create a classifier: a support vector classifier
    #classifier = svm.SVC(gamma=0.001)
    classifier = KNeighborsClassifier(n_neighbors=10)

    # Split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.1, shuffle=False)
    #X_train.to_excel('X_train.xlsx')
    #y_train.to_excel('Y_train.xlsx')

    # We learn the digits on the first half of the digits
    classifier.fit(X_train, y_train)

    # Now predict the value of the digit on the second half:
    predicted = classifier.predict(X_test)

    print('y_test', y_test)
    print('y_pred', predicted)

def demo():
    # The digits dataset
    digits = datasets.load_digits()

    # The data that we are interested in is made of 8x8 images of digits, let's
    # have a look at the first 4 images, stored in the `images` attribute of the
    # dataset.  If we were working from image files, we could load them using
    # matplotlib.pyplot.imread.  Note that each image must have the same size. For these
    # images, we know which digit they represent: it is given in the 'target' of
    # the dataset.
    _, axes = plt.subplots(2, 4)
    images_and_labels = list(zip(digits.images, digits.target))
    for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Training: %i' % label)

    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)

    # Split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False)

    # We learn the digits on the first half of the digits
    classifier.fit(X_train, y_train)

    X_train.to_excel('X_train.xlsx')
    y_train.to_excel('Y_train.xlsx')

    # Now predict the value of the digit on the second half:
    predicted = classifier.predict(X_test)

    print(predicted)

    images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
    for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Prediction: %i' % prediction)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, predicted)))
    disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)

    plt.show()


if __name__ == '__main__':

    # 画出mnist中特定数字的透射谱
    # plot_skl_mnist_result(9)
    # 根据特定几个数字的透射谱进行分类
    plot_skl_mnist_result_mean([0, 2, 4, 6, 8])