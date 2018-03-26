if __name__ == '__main__':
    import sys
    sys.path.append('/Users/hyzhang/MachineLearning/python/primary_practice')
from data import data_loader
import numpy as np
import math
import pickle
import os
import time
from scipy import signal


PARAMS_FILE = 'nn_params'
KERNEL1 = 'kernel1'
BIAS1 = 'bias1'
KERNEL2 = 'kernel2'
BIAS2 = 'bias2'
FC_WEIGHTS = 'fc_weights'
FC_BIAS = 'fc_bias'
OUTPUT_WEIGHTS = 'output_weights'
OUTPUT_BIAS = 'output_bias'
kernel1 = (np.random.random((32, 1, 3, 3)) - 0.5) / 10
bias1 = (np.random.random(28 * 28 * 32) - 0.5) / 10
kernel2 = (np.random.random((64, 32, 3, 3)) - 0.5) / 10
bias2 = (np.random.random(14 * 14 * 64) - 0.5) / 10
fc_weights = (np.random.random((1024, 7 * 7 * 64)) - 0.5) / 10
fc_bias = (np.random.random((1024, 1)) - 0.5) / 10
output_weights = (np.random.random((10, 1024)) - 0.5) / 10
output_bias = (np.random.random((10, 1)) - 0.5) / 10


def train():
    print('Try reading the params...')
    if try_read_params():
        print('Using old parameters')
    loader = data_loader.MnistLoader()
    print('Reading the data set...')
    training_set, training_labels = loader.get_training_set(True)
    print('Start trainging...')
    batch_size = 100
    count = int(len(training_labels) / batch_size)
    alpha = 0.1
    for i in range(10):
        start = time.time()
        # if i % 100:
        print(i)
        index = (i + 3) * batch_size
        images = training_set[index: index + batch_size]
        labels = training_labels[index: index + batch_size]
        # image = np.random.randint(0, 5, (1, 28, 28))
        # images = np.array([image])
        # labels = np.array([1])
        bp(images, labels, alpha)
        print('Wasted time is %d' % ((time.time() - start)/60))
        # alpha -= 0.006
    save()
    

def test():
    loader = data_loader.MnistLoader()
    test_set, test_labels = loader.get_test_set()
    count = len(test_set)
    right_count = 0
    for i in range(count):
        prediction = predict(test_set[i], test_labels[i])
        if prediction == test_labels[i]:
            right_count += 1
    print('The accuracy is %f' %(right_count / count))


def bp(images, labels, alpha):
    global kernel1
    global bias1
    global kernel2
    global bias2
    global fc_weights
    global fc_bias
    global output_weights
    global output_bias
    count = len(images)
    conv1_act_result = []
    pool1_result = []
    conv2_act_result = []
    pool2_result = []
    fc_act_result = []
    output = []
    print('Start forward propagation')
    for i in range(count):
        image = images[i]
        conv1_result = convolution(image, kernel1, bias1)
        conv1_act_result.append(ReLU(conv1_result))
        pool1_result.append(max_pool(conv1_act_result[i]))
        conv2_result = convolution(pool1_result[i], kernel2, bias2)
        conv2_act_result.append(ReLU(conv2_result))
        pool2_result.append(max_pool(conv2_act_result[i]))
        vector = pool2_result[i].reshape(7 * 7 * 64, 1)
        fc_result = np.matmul(fc_weights, vector) + fc_bias  # 1024 * 1
        fc_act_result.append(ReLU(fc_result, True))
        raw_output = np.matmul(output_weights, fc_act_result[i]) + output_bias
        output.append(softmax(raw_output))
    output_error = np.empty(10)
    error = 0
    for i in range(count):
        label = labels[i]
        error += (-math.log(output[i][label]) / count)
    print(error)
    for j in range(count):
        for i in range(10):
            output_error[i] = ((output[j][i]) + (-1 if i == labels[j] else 0)) * (1 / count)
    output_weight_gradient = []
    output_bias_gradient = []
    fc_weight_gradient = []
    fc_bias_gradient = []
    kernel2_gradient = []
    bias2_gradient = []
    kernel1_gradient = []
    bias1_gradient = []
    print('Start back propagation')
    for i in range(count):
        output_error = output_error.reshape(10, 1)  # 10 * 1
        output_weight_gradient.append(np.matmul(output_error, fc_act_result[i].T))
        output_bias_gradient.append(output_error)
        fc_error = np.matmul(output_weights.T, output_error) * get_derivative(fc_act_result[i])  # 1024 * 1
        vector = pool2_result[i].reshape(7 * 7 * 64, 1)
        fc_weight_gradient.append(np.matmul(fc_error, vector.T))
        fc_bias_gradient.append(fc_error)
        pool2_error = np.matmul(fc_weights.T, fc_error)  # (7 * 7 * 64) * 1
        conv2_error = get_conv_error(conv2_act_result[i], pool2_result[i], pool2_error) * get_derivative(conv2_act_result[i])  # 64 * 14 * 14
        kernel2_gradient.append(get_kernel_derivative(conv2_error, pool1_result[i]))  # 64 * 32 * 3 * 3
        bias2_gradient.append(conv2_error.reshape(14 * 14 * 64))
        pool1_error = back_conv(conv2_error, flip(kernel2))  # 32 * 14 * 14
        conv1_error = get_conv_error(conv1_act_result[i], pool1_result[i], pool1_error) * get_derivative(conv1_act_result[i])  # 32 * 28 * 28
        kernel1_gradient.append(get_kernel_derivative(conv1_error, images[i]))  # 32 * 1 * 28 * 28
        bias1_gradient.append(conv1_error.reshape(28 * 28 * 32))
    output_weights_g = np.mean(output_weight_gradient, axis=0)
    output_bias_g = np.mean(output_bias_gradient, axis=0)
    fc_weight_g = np.mean(fc_weight_gradient, axis=0)
    fc_bias_g = np.mean(fc_bias_gradient, 0)
    kernel2_g = np.mean(kernel2_gradient, 0)
    bias2_g = np.mean(bias2_gradient, 0)
    kernel1_g = np.mean(kernel1_gradient, 0)
    bias1_g = np.mean(bias1, 0)
    # print(kernel1_g)
    # print(kernel2_g)
    output_weights -= alpha * output_weights_g
    output_bias -= alpha * output_bias_g
    fc_weights -= alpha * fc_weight_g
    fc_bias -= alpha * fc_bias_g
    kernel2 -= alpha * kernel2_g
    bias2 -= alpha * bias2_g
    kernel1 -= alpha * kernel1_g
    bias1 -= alpha * bias1_g


def predict(image, label):
    image = image
    conv1_result = convolution(image, kernel1, bias1)
    conv1_act_result = ReLU(conv1_result)
    pool1_result = max_pool(conv1_act_result)
    conv2_result = convolution(pool1_result, kernel2, bias2)
    conv2_act_result = ReLU(conv2_result)
    pool2_result= max_pool(conv2_act_result)
    vector = pool2_result.reshape(7 * 7 * 64, 1)
    fc_result = np.matmul(fc_weights, vector) + fc_bias  # 1024 * 1
    fc_act_result = ReLU(fc_result, True)
    raw_output = np.matmul(output_weights, fc_act_result) + output_bias
    output = softmax(raw_output)
    p = 0
    index = 0
    for i in range(10):
        if output[i] > p:
            p = output[i]
            index = i
    print('The probability of \'%d\' is %f. And the true label is %d' %(index, p, label))
    return index


def get_derivative(vector):
    result = vector + 0
    vector_shape = np.shape(vector)
    for i in range(vector_shape[0]):
        for j in range(vector_shape[1]):
            if len(vector_shape) == 3:
                for k in range(vector_shape[2]):
                    result[i][j][k] = 1 if result[i][j][k] > 0 else 0
            else:
                result[i][j] = 1 if result[i][j] > 0 else 0
    return result


def get_conv_error(conv_result, pool_result, error):
    pool_shape = np.shape(pool_result)
    pool_count = pool_shape[0]
    pool_height = pool_shape[1]
    pool_width = pool_shape[2]
    e = error.reshape(pool_count, pool_height, pool_width)
    result = np.random.random((pool_count, pool_height * 2, pool_width * 2))
    for k in range(pool_count):
        for i in range(pool_height):
            for j in range(pool_width):
                for x in range(2):
                    for y in range(2):
                        result[k][i * 2 + x][j * 2 + y] = e[k][i][j] if pool_result[k][i][j] == conv_result[k][i * 2 + x][j * 2 + y] else 0
    return result


def flip(kernels):
    return np.flip(np.flip(kernels, -1), -2)


def convolution(data, kernels, bias=None):
    data_shape = np.shape(data)
    if len(data_shape) < 3:
        data = data.reshape(1, data_shape[0], data_shape[1])
        data_shape = np.shape(data)
    data_width = data_shape[2]
    data_height = data_shape[1]
    kernel_shape = np.shape(kernels)
    kernel_count = kernel_shape[0]
    kernel_len = kernel_shape[1]
    kernel_width = kernel_shape[2]
    kernel_height = kernel_shape[3]
    if True:
        conv_image = np.empty((kernel_count, data_height, data_width))
        for i in range(kernel_count):
            conv = np.zeros((data_height, data_width))
            for j in range(kernel_len):
                conv += signal.correlate2d(data[j], kernels[i][j], mode='same')
            conv_image[i] = conv
        b = 0
        if bias is not None:
            b = bias.reshape((kernel_count, data_height, data_width))
        return conv_image + b
    if kernel_len != data_shape[0]:
        raise Exception('not match')
    conv_image = np.random.random(size=(kernel_count, data_height, data_width))
    for i in range(kernel_count):
        kernel = kernels[i]
        result = 0
        temp = 0
        for x in range(data_width):
            for y in range(data_height):
                for j in range(kernel_len):
                    for k in range(kernel_width):
                        for l in range(kernel_height):
                            data_x = x - int(kernel_width / 2) + k
                            data_y = y - int(kernel_height / 2) + l
                            if data_x < 0 or data_x >= data_width or data_y < 0 or data_y >= data_height:
                                temp = 0
                            else:
                                temp = data[j][data_x][data_y]
                            result += temp * kernel[j][k][l]
                if bias is None:
                    b = 0
                else:
                    b = bias[i * data_width * data_height + x * data_width + y]
                conv_image[i][x][y] = result + b
                result = 0
    return conv_image


def back_conv(error, flipped_kernel):
    kernel_shape = np.shape(flipped_kernel)
    kernel_count = kernel_shape[0]
    kernel_channel = kernel_shape[1]
    kernel_height = kernel_shape[2]
    kernel_width = kernel_shape[3]
    kernel = np.random.random((kernel_channel, kernel_count, kernel_height, kernel_width))
    for i in range(kernel_count):
        for j in range(kernel_channel):
            kernel[j][i] = flipped_kernel[i][j]
    return convolution(error, kernel)


def get_kernel_derivative(error, output):
    error_shape = np.shape(error)
    output_shape = np.shape(output)
    if len(output_shape) < 3:
        output = output.reshape(1, output_shape[0], output_shape[1])
        output_shape = np.shape(output)
    count = error_shape[0]
    channel = output_shape[0]
    if error_shape[1] != output_shape[1] or error_shape[2] != output_shape[1]:
        raise Exception('Not match!')
    result = np.empty((count, channel, 3, 3))
    for i in range(count):
        for j in range(channel):
            for x in range(-1, 2):
                for y in range(-1, 2):
                    kernel = shift(output[j], x, y)
                    result[i][j][x+1][y+1] = signal.convolve2d(error[j], kernel, 'valid')
                    # result[i][j][x+1][y+1] = conv_2d(error[j], kernel)
    return result


def shift(output, x, y):
    kernel = np.roll(np.roll(output, x, 1), y, 0) + 0
    if x == -1:
        kernel[:, 0] *= 0
    elif x == 1:
        kernel[:, -1] *= 0
    if y == -1:
        kernel[0] *= 0
    elif y == 1:
        kernel[-1] *= 0
    return kernel


# return a scalar, and will flip
def conv_2d(data, kernel):
    width = np.shape(data)[1]
    height = np.shape(data)[0]
    k = flip(kernel)
    result = 0
    for i in range(width):
        for j in range(height):
            result += (k[i][j] * data[i][j])
    return result


def max_pool(data):
    data_shape = np.shape(data)
    data_width = data_shape[2]
    data_height = data_shape[1]
    data_len = data_shape[0]
    pooling_image = np.random.random(size=(data_len, int(data_height / 2), int(data_width / 2)))
    for i in range(data_len):
        for x in range(int(data_width / 2)):
            for y in range(int(data_height / 2)):
                pooling_image[i][x][y] = max(data[i][x * 2][y * 2],
                                             data[i][x * 2 + 1][y * 2],
                                             data[i][x * 2][y * 2 + 1],
                                             data[i][x * 2 + 1][y * 2 + 1])
    return pooling_image


def ReLU(image, is_panel=False):
    image_shape = np.shape(image)
    if is_panel:
        image_width = image_shape[0]
        image_height = image_shape[1]
        result = np.random.random((image_width, image_height))
        for i in range(image_width):
            for j in range(image_height):
                result[i][j] = max(0, image[i][j])
        return result
    map_count = image_shape[0]
    image_height = image_shape[1]
    image_width = image_shape[2]
    result = np.empty((map_count, image_height, image_width))
    for i in range(map_count):
        for j in range(image_height):
            for k in range(image_width):
                result[i][j][k] = max(0, image[i][j][k])
    return result


def softmax(vector):
    vector_shape = np.shape(vector)
    result = vector.reshape(vector_shape[0] * vector_shape[1])
    maxima = max(result)
    result -= maxima
    count = len(result)
    su = 0
    for i in range(count):
        result[i] = math.pow(math.e, result[i])
        su += result[i]
    for i in range(count):
        result[i] = result[i] / su
    return result


def save():
    params = {KERNEL1: kernel1,
              BIAS1: bias1,
              KERNEL2: kernel2,
              BIAS2: bias2,
              FC_WEIGHTS: fc_weights,
              FC_BIAS: fc_bias,
              OUTPUT_WEIGHTS: output_weights,
              OUTPUT_BIAS: output_bias}
    f = open(PARAMS_FILE, 'wb')
    pickle.dump(params, f)
    f.close()


def try_read_params():
    if not os.path.exists('./' + PARAMS_FILE):
        return False
    global kernel1
    global bias1
    global kernel2
    global bias2
    global fc_weights
    global fc_bias
    global output_weights
    global output_bias
    f = open(PARAMS_FILE, 'rb')
    params = pickle.load(f)
    kernel1 = params[KERNEL1]
    bias1 = params[BIAS1]
    kernel2 = params[KERNEL2]
    bias2 = params[BIAS2]
    fc_weights = params[FC_WEIGHTS]
    fc_bias = params[FC_BIAS]
    output_weights = params[OUTPUT_WEIGHTS]
    output_bias = params[OUTPUT_BIAS]
    return True


def function_test():
    data = [[[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]],
            [[-1, -2, -3],
             [-4, -5, -6],
             [-7, -8, -9]]]
    data = [[[1, 2, 3, 1],
             [4, 5, 6, 4],
             [7, 8, 9, 7],
             [10, 11, 12, 10]]]
    data = np.array(data)
    kernel = [[[[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]],
               [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]]]
    kernel = [[[[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]]]
    bias = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16]
    kernel = np.array(kernel)
    t = convolution(data, kernel, bias)
    print(t)
    print(max_pool(t))

if __name__ == '__main__':
    # image = np.random.randint(0, 5, (28, 28))
    # start = time.time()
    # bp([image], [1], 0.1)
    # bp([image], [1], 0.1)
    # bp([image], [1], 0.1)
    # print(time.time() - start)
    # train()
    try_read_params()
    # test()
    test_set, test_labels = data_loader.MnistLoader().get_test_set(True)
    images = test_set[0: 30]
    labels = test_labels[0: 30]
    bp(images, labels, 0.1)
    bp(images, labels, 0.1)
    bp(images, labels, 0.1)
    bp(images, labels, 0.1)
