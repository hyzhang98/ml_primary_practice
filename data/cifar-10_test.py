import pickle
import numpy
import matplotlib.pyplot as plot
import os

def main():
    path = '/Users/hyzhang/MachineLearning/python/primary_practice/data/cifar-10-batches-py'
    data_path = os.path.join(path, 'data_batch_1')
    data_file = open(data_path, 'rb')
    data_dict = pickle.load(data_file, encoding='iso-8859-1')
    print(data_dict.keys())
    data = numpy.array(data_dict.get('data'))
    red = []
    for i in range(32 * 32):
        red.append(data[0][i])
    red = numpy.array(red).reshape(1024, 1)
    blue = []
    for i in range(32 * 32):
        blue.append(data[0][i + 1024])
    blue = numpy.array(blue).reshape(1024, 1)
    green = []
    for i in range(32 * 32):
        green.append(data[0][i + 1024 * 2])
    green = numpy.array(green).reshape(1024, 1)
    pic0 = numpy.hstack((red, blue, green))
    print(type(pic0))
    pic0 = pic0.reshape(32, 32, 3)
    print(data_dict.get('labels')[0])
    plot.imshow(pic0)
    #plot.show()
    data_file.close()


if __name__ == '__main__':
    main()

