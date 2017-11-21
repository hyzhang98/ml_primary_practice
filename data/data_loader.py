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

    def get_training_set(self):
        images = self._get_images_from_file(os.path.join(self.base_path, 'mnist/mnist-train-images-ubyte'))
        labels = self._get_labels_from_file(os.path.join(self.base_path, 'mnist/mnist-train-labels-ubyte'))
        if not len(images) == len(labels):
            print('The count of images is '), len(images)
            print('The count of labels is '), len(labels)
            raise Exception('The count of images doesn\'t equals with labels!')
        return images, labels

    
    def get_test_set(self):
        images = self._get_images_from_file('./mnist/mnist-test-10k-images-ubyte')
        labels = self._get_labels_from_file('./mnist/mnist-test-10k-labels-ubyte')
        if not len(images) == len(labels):
            print('The count of test-images is '), len(images)
            print('The count of test-labels is '), len(labels)
            raise Exception('The count of test-images doesn\'t equals with test-labels!')
        return images, labels


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


