import numpy as np
import pickle
from data import data_loader
from scipy import stats


class CART:
    def __init__(self, training_set, features, params_file_name='CART_tree'):
        self.training_set = training_set
        self.root = None
        self.training_set_size = len(training_set)
        self.params_file_name = params_file_name
        self.features = features

    def train(self):
        self.build(None, self.features, self.training_set, True, 10)
        f = open(self.params_file_name, 'wb')
        pickle.dump(self.root, f)
        f.close()

    def build(self, parent, features, data_set, left, category_count):
        self_gini = self.cal_gini(data_set, category_count)
        if self._should_stop(self_gini, data_set, features):
            parent.label = self.max_num_label(data_set, category_count)
            return
        gini, feature = self.get_minimum_gini_feature(features, data_set, category_count)
        if gini > 0.95:
            parent.label = self.max_num_label(data_set, category_count)
            return
        # grow
        p = None
        if parent is None:
            self.root = Node(feature)
            p = self.root
        else:
            if left:
                parent.left = Node(feature)
                p = parent.left
            else:
                parent.right = Node(feature)
                p = parent.right
        #  del features and split data_set
        new_features = features[:]
        new_features.remove(feature)
        data_set1 = []
        data_set2 = []
        for data in data_set:
            if data[0][feature] == 0:
                data_set1.append(data)
            else:
                data_set2.append(data)
        # if not data_set1:
        #     parent.label =
        self.build(p, new_features, data_set1, True, category_count)
        self.build(p, new_features, data_set2, False, category_count)

    def max_num_label(self, data_set, category_count):
        nums = np.zeros(category_count)
        for data in data_set:
            label = data[1]
            nums[label] += 1
        return nums.argmax()

    def get_minimum_gini_feature(self, features, data_set, category_count):
        data_set_size = len(data_set)
        ginis = []
        for feature in features:
            data_set1 = []
            data_set2 = []
            for data in data_set:
                if data[0][feature] == 1:
                    data_set1.append(data)
                else:
                    data_set2.append(data)
            gini1 = self.cal_gini(data_set1, category_count)
            gini2 = self.cal_gini(data_set2, category_count)
            if gini1 == 1 or gini2 == 1:
                gini = 1
            else:
                gini = (len(data_set1) / data_set_size) * gini1 + \
                            (len(data_set2) / data_set_size) * gini2
            ginis.append((gini, feature))
        mini_gini = min(ginis, key=lambda k: k[0])
        return mini_gini[0], mini_gini[1]

    def cal_ginis(self):
        pass

    def cal_gini(self, data_set, category_count):
        p = np.zeros(category_count)
        count = len(data_set)
        if count == 0:
            return 1
        for data in data_set:
            label = data[1]
            p[label] += 1
        p /= count
        p *= p
        return 1 - np.sum(p)

    def _should_stop(self, gini, data_set, features):
        if len(data_set) == 0:
            print('---1---')
            return True
        if gini <= 0.01:
            # print('----2----', end=' ')
            # print(gini)
            return True
        # if len(data_set) / self.training_set_size < 0.05:
        #     return True
        if not features:
            return True
        return False

    def predict(self, sample):
        return self.root.predict(sample)


class Node:
    def __init__(self, feature, label=None):
        self.feature = feature
        self.left = None
        self.right = None
        self.label = label

    def predict(self, value):
        if value[self.feature] == 0:
            if self.left is None:
                return self.label
            return self.left.predict(value)
        else:
            if self.right is None:
                return self.label
            return self.right.predict(value)


def test():
    loader = data_loader.MnistLoader()
    data_set = loader.get_united_test_set(True)
    count = len(data_set)
    for i in range(count):
        data_set[i] = (data_set[i][0].reshape(28*28), data_set[i][1])
    tree = CART(data_set[0:8000])
    tree.train()
    right = 0
    for i in range(8000):
        if tree.predict(data_set[i][0]) == data_set[i][1]:
            right += 1
    print(right)
    right = 0
    for i in range(2000):
        if tree.predict(data_set[i+8000][0]) == data_set[i+8000][1]:
            right += 1
    # print(tree.predict(data_set[0][0]))
    # print(tree.predict(data_set[1][0]))
    # print(tree.predict(data_set[2][0]))
    print(right)


if __name__ == '__main__':
    loader = data_loader.MnistLoader()
    data_set = loader.get_united_training_set(True)
    count = len(data_set)
    for i in range(count):
        data_set[i] = (data_set[i][0].reshape(28 * 28), data_set[i][1])
    tree = CART(data_set, list(range(0, 28*28)))
    # tree.train()
    f = open('CART_tree', 'rb')
    root = pickle.load(f)
    tree.root = root
    right = 0
    for i in range(count):
        if tree.predict(data_set[i][0]) == data_set[i][1]:
            right += 1
    print(right)
    test_set = loader.get_united_test_set(True)
    count = len(test_set)
    for i in range(count):
        test_set[i] = (test_set[i][0].reshape(28 * 28), test_set[i][1])
    right = 0
    for i in range(count):
        if tree.predict(test_set[i][0]) == test_set[i][1]:
            right += 1
    print(right)
