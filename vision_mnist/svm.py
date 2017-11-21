if __name__ == '__main__':
    import sys
    sys.path.append('/Users/hyzhang/MachineLearning/python/primary_practice')
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math.exp
from data.data_loader import MnistLoader


class SupportVectorMachine:
    def __init__(self):
        self.SOFT_LIMIT = 10
        self.sigma = 1
        self._get_training_set()
        self.training_set_size = len(self.train_images)
        self._init_params()
        self._init_kernel_value()

    def _get_training_set(self):
        self.train_images, self.train_labels = MnistLoader().get_training_set()

    def _init_params(self):
        self.alphas = []
        for i in range(self.training_set_size):
            if i < self.training_set_size - 1:
                self.alphas.append(random.uniform(0, self.SOFT_LIMIT))
            else:
                self.alphas.append(self._get_final_alpha())

    def _get_final_alpha(self):
        training_sum = 0
        for i in range(len(self.alphas)):
            flag = -1
            if self.train_labels[i] == 0:
                flag = 1
            training_sum += flag * self.alphas[i]
        return training_sum * -1

    def _init_kernel_value(self):
        self.kernel_value = []
        for i in range(self.training_set_size):
            item = []
            for j in range(self.training_set_size):
                item.append(-1)
            self.kernel_value.append(item)

    def start(self):
        self.start_SMO()

    def start_SMO(self):
        while (True):
            index1, index2 = self._choose_two_alphas()
            alpha = self.alphas[index1]
            beta = self.alphas[index2]
            break

    def _choose_two_alphas(self):
        index1 = random.randint(0, self.training_set_size - 1)
        index2 = random.randint(0, self.training_set_size - 1)
        index2 = index2 if index1 != index2 else index - 1
        if index1 == index2:
            index2 = index2 - 1 if index2 != 0 else index2 + 1
        return index1, index2

    def _kernel_function(index1, index2):
        value = kernel_value[index1][index2]
        if value >= 0:
            return value;
        image_alpha = self.train_images[index1]
        image_beta = self.train_images[index2]
        vector_diff = image_alpha - image_beta;
        multi = np.dot(vector_diff, vector_diff)
        value = exp(vector_diff / (2 * (self.sigma ** 2))* -1)
        self.kernel_value[index1][index2] = value
        self.kernel_value[index2][index1] = value


def train():
    start = time.time()
    print('Reading Training Set...')
    svm = SupportVectorMachine()
    svm.start()
    end = time.time()
    print('End within ' + str(end - start) + 's')


if __name__ == '__main__':
    train()

