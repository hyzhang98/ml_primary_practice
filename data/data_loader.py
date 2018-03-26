import struct
import numpy as np
import os

class MnistLoader:
    def __init__(self):
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.base_path = os.path.dirname(__file__)

    def _is_in_image_cache(self, is_train):
        return self.train_images if is_train else self.test_images

    def _is_in_labels_cache(self, is_train):
        return self.train_labels if is_train else self.test_labels

    def get_training_set(self, grayscale=False):
        if self.train_images and self.train_labels:
            return self.train_images, self.train_labels
        self.train_images = self._get_images_from_file(os.path.join(self.base_path, 'mnist/mnist-train-images-ubyte'))
        self.train_labels = self._get_labels_from_file(os.path.join(self.base_path, 'mnist/mnist-train-labels-ubyte'))
        if not len(self.train_images) == len(self.train_labels):
            print('The count of images is '), len(self.train_images)
            print('The count of labels is '), len(self.train_labels)
            raise Exception('The count of images doesn\'t equals with labels!')
        if not grayscale:
            return self.train_images, self.train_labels
        else:
            for image in self.train_images:
                shape = np.shape(image)
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        # image[i][j] = 0 if image[i][j] == 0 else 1
                        value = image[i][j]
                        if value == 255 or value == 0:
                            value -= 1
                        image[i][j] = (int)(85 + value) / 85
        return self.train_images, self.train_labels

    def get_united_training_set(self, binary=False):
        train_images = self._get_images_from_file(os.path.join(self.base_path, 'mnist/mnist-train-images-ubyte'))
        train_labels = self._get_labels_from_file(os.path.join(self.base_path, 'mnist/mnist-train-labels-ubyte'))
        if not len(self.train_images) == len(self.train_labels):
            print('The count of images is '), len(self.train_images)
            print('The count of labels is '), len(self.train_labels)
            raise Exception('The count of images doesn\'t equals with labels!')
        if binary:
            shape = np.shape(train_images[0])
            for image in train_images:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        if image[i][j] > 0:
                            image[i][j] = 1
        count = len(train_labels)
        result = []
        for i in range(count):
            result.append((train_images[i], train_labels[i]))
        return result

    def get_test_set(self, grayscale=False):
        if self.test_images and self.test_labels:
            return self.test_images, self.test_labels
        self.test_images = self._get_images_from_file(os.path.join(self.base_path, 'mnist/mnist-test-10k-images-ubyte'))
        self.test_labels = self._get_labels_from_file(os.path.join(self.base_path, 'mnist/mnist-test-10k-labels-ubyte'))
        if not len(self.test_images) == len(self.test_labels):
            print('The count of test-images is '), len(self.test_images)
            print('The count of test-labels is '), len(self.test_labels)
            raise Exception('The count of test-images doesn\'t equals with test-labels!')
        if not grayscale:
            return self.test_images, self.test_labels
        else:
            for image in self.test_images:
                shape = np.shape(image)
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        # image[i][j] = 0 if image[i][j] == 0 else 1
                        value = image[i][j]
                        if value == 255 or value == 0:
                            value -= 1
                        image[i][j] = (int)(85 + value) / 85
        return self.test_images, self.test_labels

    def get_united_test_set(self, binary=False):
        test_images = self._get_images_from_file(os.path.join(self.base_path, 'mnist/mnist-test-10k-images-ubyte'))
        test_labels = self._get_labels_from_file(os.path.join(self.base_path, 'mnist/mnist-test-10k-labels-ubyte'))
        if not len(self.test_images) == len(self.test_labels):
            print('The count of test-images is '), len(self.test_images)
            print('The count of test-labels is '), len(self.test_labels)
            raise Exception('The count of test-images doesn\'t equals with test-labels!')
        if binary:
            shape = np.shape(test_images[0])
            for image in test_images:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        if image[i][j] > 0:
                            image[i][j] = 1
        count = len(test_labels)
        result = []
        for i in range(count):
            result.append((test_images[i], test_labels[i]))
        return result

    def get_all_set(self):
        train_images, train_labels = self.get_training_set()
        test_images, test_labels = self.get_test_set()
        images = []
        images.extend(train_images)
        images.extend(test_images)
        labels = []
        labels.extend(train_labels)
        labels.extend(test_labels)
        return images, labels

    def _get_images_from_file(self, image_path):
        image_file = open(image_path, 'rb')
        magic, count, rows, columns = struct.unpack('>IIII', image_file.read(16))
        images = []
        size = rows * columns
        unpack_mode = str(size) + 'B'
        for i in range(count):
            image = image_file.read(size)
            image = struct.unpack(unpack_mode, image)
            image = np.array(image).reshape(rows, columns)
            images.append(image)
        image_file.close()
        return images

    def _get_labels_from_file(self, label_path):
        label_file = open(label_path, 'rb')
        magic, count = struct.unpack('>II', label_file.read(8))
        labels = []
        for i in range(count):
            labels.append(struct.unpack('1B', label_file.read(1))[0])
        label_file.close()
        return labels


DEFAULT_LOADER = MnistLoader()

