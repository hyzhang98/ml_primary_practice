if __name__ == '__main__':
    import sys
    sys.path.append('/Users/hyzhang/MachineLearning/python/primary_practice')
import matplotlib.pyplot as plt
import numpy
import random
import pickle
import time
import math.exp
from data.data_loader import MnistLoader


class SupportVectorMachine:
    def __init__(self):
        self.PARAMS_FILE = 'svm_params'
        self.SOFT_LIMIT = 10
        self.ERROR = 0.001
        self.sigma = 1
        self._get_training_set()
        self.training_set_size = len(self.train_images)
        self._init_params()
        self._init_kernel_value()
        self.b = 0

    def _get_training_set(self):
        self.train_images, self.train_labels = MnistLoader().get_training_set()

    def _init_params(self):
        self.alphas = []
        for i in range(self.training_set_size):
            self.alphas.append(0)

    def _init_kernel_value(self):
        self.kernel_value = []
        for i in range(self.training_set_size):
            item = []
            for j in range(self.training_set_size):
                item.append(-1)
            self.kernel_value.append(item)

    def start(self):
        alphas, b = self._try_read_params()
        if alphas:
            self.alphas = alphas
            self.b = b
        self.start_SMO()

    def start_SMO(self):
        for i in range(self.training_set_size):
            if not self._is_obey_KKT(i):
                index1 = i
                index2 = self._choose_second_alphas(index2)
                self.step(index1, index2)
        self.non_bound_set = []
        for i in range(self.training_set_size):
            if 0 < self.alphas[i] < self.SOFT_LIMIT and not self._is_obey_KKT(i):
                self.non_bound_set.append(i)
        while self.non_bound_set:
            index1 = random.choice(random)
            index2 = self._choose_second_alphas(index1)
            if self._is_obey_KKT(index1) and 0 < self.alphas[index1] < self.SOFT_LIMIT:
                non_bound_set.remove(non_bound_set.index(index1))
            if not self._is_obey_KKT(index2) and 0 < self.alphas[index2] < self.SOFT_LIMIT:
                non_bound_set.append(index2)

    def _is_obey_KKT(self, index):
        y = self._get_label(index)
        result = y * self._get_training_set_predict_value(index)
        alpha = self.alphas[index]
        if alpha == 0:
            return result >= 1
        elif alpha == self.SOFT_LIMIT:
            return result <= 1
        elif 0 < alpha < self.SOFT_LIMIT:
            return 1 + self.ERROR + result < 1 + self.ERROR
        else:
            return False

    def _choose_second_alphas(self, index):
        e1 = self._get_error(index)
        result = 0
        t = 0
        if e1 > 0:
            for i in range(self.training_set_size):
                e2 = self._get_error(i)
                if e2 > t:
                    t = e2
                    result = i
        else:
            for i in range(self.training_set_size):
                e2 = self._get_error(i)
                if e2 < t:
                    t = e2
                    result = i
        return result

    def _get_error(self, index):
        return self._get_training_set_predict_value(index) - self._get_error(index)

    def step(self, index1, index2):
        alpha1 = self.alphas[index1]
        alpha2 = self.alphas[index2]
        y1 = self._get_label(index1)
        y2 = self._get_label(index2)
        e1 = self._get_error(index1)
        e2 = self._get_error(index2)
        L = 0
        H = C
        if y1 == y2:
            L = max(0, alpha2 + alpha1 - C)
            H = min(C, alpha2 + alpha1)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(C, alpha2 - alpha1 + C)
        eta = self._K(index1, index1) + self._K(index2, index2) - 2 * self._K(index1, index2)
        new_alpha2 = alpha2 + y2 * (e1 - e2) / eta
        if new_alpha2 >= H:
            new_alpha2 = H
        elif new_alpha2 <= L:
            new_alpha2 = L
        new_alpha1 = alpha1 + y1 * y2 * (alpha2 - new_alpha2)
        self.alphas[index1] = new_alpha1
        self.alphas[index2] = new_alpha2
        b1 = e1 + y1 * (new_alpha1 - alpha1) * self._K(index1, index1) + y2 * (new_alpha2 - alpha2) * self._K(index1, index2) + b
        b2 = e2 + y1 * (new_alpha1 - alpha1) * self._K(index1, index2) + y2 * (new_alpha2 - alpha2) * self._K(index2, index2) + b
        if 0 < alpha1 < self.SOFT_LIMIT:
            self.b = b1
        elif 0  < alpha2 < self.SOFT_LIMIT:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

    def _K(self, index1, index2):
        value = kernel_value[index1][index2]
        if value >= 0:
            return value;
        image_alpha = self.train_images[index1]
        image_beta = self.train_images[index2]
        value = self._gaussian_kernel(image_alpha, image_beta)
        self.kernel_value[index1][index2] = value
        self.kernel_value[index2][index1] = value
        return value

    def _kernel_function(self, image_alpha, image_beta):
        return self._gaussian_kernel(image_alpha, image_beta)

    def _gaussian_kernel(self, image_alpha, image_beta):
        vector_diff = image_alpha - image_beta;
        multi = np.dot(vector_diff, vector_diff)
        return exp(vector_diff / (2 * (self.sigma ** 2))* -1)

    def _get_label(self, index):
        return 1 if self.train_labels[index] == 0 else -1

    def _get_training_set_predict_value(self, index):
        result = 0
        for i in range(self.training_set_size):
            if self.alphas[i] == 0:
                continue
            result += self.alphas[i] * self._get_label(i) * self._K(i, index)
        return result

    def predict(self, sample):
        result = 0
        for i in range(self.training_set_size):
            if self.alphas[i] == 0:
                continue
            result += self.alphas[i] * self._get_label(i) * self._kernel_function(self.train_images[i], sample)
        return result + self.b

    def _save_params(self):
        obj = [self.alphas, self.b]
        f = open(self.PARAMS_FILE, 'wb')
        pickle.dump(obj, f)
        f.close()

    def _try_read_params(self):
        obj = []
        try:
            f = open(self.PARAMS_FILE, 'rb')
            obj = pickle.load(f)
            f.close()
        except:
            obj = [[], 0]
        return obj[0], obj[1]


def train():
    start = time.time()
    print('Reading Training Set...')
    svm = SupportVectorMachine()
    svm.start()
    end = time.time()
    print('End within ' + str(end - start) + 's')


if __name__ == '__main__':
    train()

