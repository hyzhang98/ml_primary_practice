if __name__ == '__main__':
    import sys
    sys.path.append('/Users/hyzhang/MachineLearning/python/primary_practice')
import matplotlib.pyplot as plt
import numpy
import random
import pickle
import time
from math import exp
from data.data_loader import *


class SupportVectorMachine:
    def __init__(self, loader, main_label=0):
        print('%d: start init...' % main_label)
        self.non_bound_set = []
        self._support_vectors = []
        self.loader = loader
        self.PARAMS_FILE = 'svm_params/svm_params' + str(main_label)
        self.SOFT_LIMIT = 10
        self.main_label = main_label
        self.ERROR = 0.001
        self.sigma = 28
        self._get_training_set()
        self.training_set_size = len(self.train_images)
        self._init_params()
        # self._init_error_cache()加快速度
        self.b = 0

    def _get_training_set(self):
        self.train_images, self.train_labels = self.loader.get_training_set(True)
        count = len(self.train_images)
        shape = numpy.shape(self.train_images[0])
        size = shape[0] * shape[1]
        for i in range(count):
            image = self.train_images[i]
            self.train_images[i] = image.reshape(size, 1)

    def _init_params(self):
        self.alphas = []
        for i in range(self.training_set_size):
            self.alphas.append(0)

    def _init_error_cache(self):
        self.error_cache = []
        for i in range(self.training_set_size):
            self.error_cache.append(-self._get_label(i))

    def start(self):
        alphas, b = self._try_read_params()
        if alphas:
            self.alphas = alphas
            self.b = b
            return
        self.start_SMO()
        self._save_params()
        # del self.error_cache[:]

    def start_SMO(self):
        print('%d: start SMO...' % self.main_label)
        for i in range(self.training_set_size):
            if not self._is_obey_KKT(i):
                index1 = i
                index2 = self._choose_second_alphas(index1)
                self.step(index1, index2)
                if not self._is_obey_KKT(index1):
                    if index1 not in self.non_bound_set:
                        self.non_bound_set.append(index1)
                else:
                    if index1 in self.non_bound_set:
                        self.non_bound_set.remove(index1)
                # if not self._is_obey_KKT(index2) and 0 < self.alphas[index2] < self.SOFT_LIMIT:
                if not self._is_obey_KKT(index2):
                    if index2 not in self.non_bound_set:
                        self.non_bound_set.append(index2)
                else:
                    if index2 in self.non_bound_set:
                        self.non_bound_set.remove(index2)
            if i % 1000 == 0:
                print('%d: first loop at index %d' % (self.main_label, i))
        print('%d: start alphas\' loop...' % self.main_label)
        while self.non_bound_set:
            index1 = random.choice(self.non_bound_set)
            index2 = self._choose_second_alphas(index1)
            self.step(index1, index2)
            if self._is_obey_KKT(index1):
                self.non_bound_set.remove(index1)
            # if not self._is_obey_KKT(index2) and 0 < self.alphas[index2] < self.SOFT_LIMIT:
            if not self._is_obey_KKT(index2) and index2 not in self.non_bound_set:
                self.non_bound_set.append(index2)
            count = len(self.non_bound_set)
            if count % 500 == 0:
                print('%d: The size of non-bound-set is %d, b is %.6f, index1 is %d, index2 is %d' % (self.main_label, len(self.non_bound_set), self.b, index1, index2))

    def _is_obey_KKT(self, index):
        y = self._get_label(index)
        result = y * self._get_training_set_predict_value(index)
        alpha = self.alphas[index]
        if alpha == 0:
            return result >= 1
        elif alpha == self.SOFT_LIMIT:
            return result <= 1
        elif 0 < alpha < self.SOFT_LIMIT:
            return 1 - self.eERROR < result < 1 + self.ERROR
        else:
            return False

    def _choose_second_alphas(self, index):
        result = index
        while result == index:
            result = random.randint(0, self.training_set_size - 1)
        return result

    def _get_error(self, index):
        return self._get_training_set_predict_value(index) - self._get_label(index)
        # return self.error_cache[index]

    def step(self, index1, index2):
        alpha1 = self.alphas[index1]
        alpha2 = self.alphas[index2]
        y1 = self._get_label(index1)
        y2 = self._get_label(index2)
        e1 = self._get_error(index1)
        e2 = self._get_error(index2)
        C = self.SOFT_LIMIT
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
        b1 = e1 + y1 * (new_alpha1 - alpha1) * self._K(index1, index1) + y2 * (new_alpha2 - alpha2) * self._K(index1, index2) + self.b
        b2 = e2 + y1 * (new_alpha1 - alpha1) * self._K(index1, index2) + y2 * (new_alpha2 - alpha2) * self._K(index2, index2) + self.b
        old_b = self.b
        if 0 < alpha1 < self.SOFT_LIMIT:
            self.b = b1
        elif 0 < alpha2 < self.SOFT_LIMIT:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        # 加速
        # for i in range(self.training_set_size):
        #     error = self.error_cache[i] + y1 * self._K(index1, i) * (new_alpha1 - alpha1) + \
        #             y2 * self._K(index2, i) * (new_alpha2 - alpha2) - self.b + old_b
        #     self.error_cache[i] = error
        #     del error
            # self.error_cache[i] = self._get_training_set_predict_value(i) - self._get_label(i)

    def _K(self, index1, index2):
        image_alpha = self.train_images[index1]
        image_beta = self.train_images[index2]
        value = self._gaussian_kernel(image_alpha, image_beta)
        return value

    def _kernel_function(self, image_alpha, image_beta):
        return self._gaussian_kernel(image_alpha, image_beta)

    def _gaussian_kernel(self, image_alpha, image_beta):
        vector_diff = image_alpha - image_beta
        multi = numpy.dot(vector_diff.T, vector_diff)[0][0]
        return exp(multi / (2 * (self.sigma ** 2)) * -1)

    def _get_label(self, index):
        return 1 if self.train_labels[index] == self.main_label else -1

    def _get_training_set_predict_value(self, index):
        result = 0
        for i in range(self.training_set_size):
            if self.alphas[i] == 0:
                continue
            result += self.alphas[i] * self._get_label(i) * self._K(i, index)
        return result - self.b

    def predict(self, sample):
        result = 0
        if not self._support_vectors:
            for i in range(self.training_set_size):
                if self.alphas[i] != 0:
                    self._support_vectors.append(i)
        count = len(self._support_vectors)
        for i in range(count):
            # if self.alphas[i] == 0:
            #     continue
            index = self._support_vectors[i]
            result += self.alphas[index] * self._get_label(index) * self._kernel_function(self.train_images[index],
                                                                                          sample)
        return result - self.b

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


def train_and_test():
    start = time.time()
    print('Reading Training Set...')
    svms = []
    loader = DEFAULT_LOADER
    for main_label in range(10):
        svm = SupportVectorMachine(loader, main_label)
        svm.start()
        svms.append(svm)
    print('End training')
    # 统计训练误差
    # print('Calculating training error...')
    # train_images, train_labels = loader.get_training_set(True)
    # count = len(train_images)
    # shape = numpy.shape(train_images[0])
    # size = shape[0] * shape[1]
    # right_count = 0
    # for i in range(count):
    #     image = train_images[i]
    #     train_images[i] = image.reshape(size, 1)
    # for i in range(count):
    #     if i % 100 == 0:
    #         print(i)
    #     image = train_images[i]
    #     result = predict(svms, image)
    #     if result == train_labels[i]:
    #         right_count += 1
    # print('The size of Train-Set is %d, and the count of right predictions is %d. The accuracy rate is %.2f%%' % (count, right_count, right_count * 100 / count))
    print('Testing...')
    test_images, test_labels = loader.get_test_set(True)
    count = len(test_images)
    shape = numpy.shape(test_images[0])
    size = shape[0] * shape[1]
    right_count = 0
    for i in range(count):
        image = test_images[i]
        test_images[i] = image.reshape(size, 1)
    for i in range(count):
        if i % 500 == 0:
            print(i)
        image = test_images[i]
        result = predict(svms, image)
        if result == test_labels[i]:
            right_count += 1
    end = time.time()
    print('The size of Test-Set is %d, and the count of right predictions is %d. The accuracy rate is %.2f%%' % (count, right_count, right_count * 100 / count))
    print('End within %.2f min' % ((end - start) / 60))


def predict(svms, sample):
    dist = svms[0].predict(sample)
    result = 0
    for i in range(10):
        if i == 0:
            continue
        label = svms[i].predict(sample)
        if label > dist:
            dist = label
            result = i
    return result

if __name__ == '__main__':
    train_and_test()

