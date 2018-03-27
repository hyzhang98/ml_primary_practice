from vision_mnist.trees import CART
from data import data_loader
import random
import numpy as np
import time
import pickle


class RandomForest:
    def __init__(self, trees_count=10, feature_rate=0.6):
        self.trees_count = trees_count
        self.feature_rate = feature_rate
        self.features = list(range(28 * 28))
        self.PARAMS_PATH = 'random_forest_params/'
        self.loader = data_loader.MnistLoader()
        self.data_set = None
        self.forest = []

    def train(self):
        if self.data_set is None:
            data_set = self.loader.get_united_training_set(True)
            count = len(data_set)
            for i in range(count):
                data_set[i] = (data_set[i][0].reshape(28 * 28), data_set[i][1])
            self.data_set = data_set
        for i in range(self.trees_count):
            print(i)
            start = time.time()
            data_set = self._generate_random_data_set()
            features = self._get_random_features()
            file_name = self.PARAMS_PATH + 'tree_' + str(i)
            tree = CART(data_set, features, file_name)
            tree.train()
            self.forest.append(tree)
            print((time.time() - start) / 60)

    def _generate_random_data_set(self):
        data_set = []
        size = len(self.data_set)
        for i in range(size):
            data_set.append(random.choice(self.data_set))
        return data_set

    def _get_random_features(self):
        feature_count = len(self.features)
        count = int(feature_count * self.feature_rate)
        return random.sample(range(0, feature_count), count)

    def test_on_test_set(self):
        print('Testing...')
        test_set = self.loader.get_united_test_set(True)
        count = len(test_set)
        for i in range(count):
            test_set[i] = (test_set[i][0].reshape(28 * 28), test_set[i][1])
        right = 0
        for data_couple in test_set:
            prediction = np.zeros(10)
            for i in range(self.trees_count):
                prediction[self.forest[i].predict(data_couple[0])] += 1
            if prediction.argmax() == data_couple[1]:
                right += 1
        print('The accuracy is %f ' % (right / len(test_set)))


if __name__ == '__main__':
    forest = RandomForest()
    # forest.train()
    for i in range(10):
        f = open('random_forest_params/tree_' + str(i), 'rb')
        root = pickle.load(f)
        tree = CART([], [])
        tree.root = root
        forest.forest.append(tree)
    forest.test_on_test_set()
