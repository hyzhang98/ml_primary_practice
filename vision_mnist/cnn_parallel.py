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
from multiprocessing import Pool

PARAMS_FILE = 'nn_params'
KERNEL1 = 'kernel1'
BIAS1 = 'bias1'
KERNEL2 = 'kernel2'
BIAS2 = 'bias2'
FC_WEIGHTS = 'fc_weights'
FC_BIAS = 'fc_bias'
OUTPUT_WEIGHTS = 'output_weights'
OUTPUT_BIAS = 'output_bias'
kernel1 = (np.random.random((32, 1, 3, 3)) - 0.5) * 2 / 3
bias1 = (np.random.random(28 * 28 * 32) - 0.5) * 2 / 3
kernel2 = (np.random.random((64, 32, 3, 3)) - 0.5) * 2 / 3
bias2 = (np.random.random(14 * 14 * 64) - 0.5) * 2 / 3
fc_weights = (np.random.random((1024, 7 * 7 * 64)) - 0.5) * 2 / 3
fc_bias = (np.random.random((1024, 1)) - 0.5) * 2 / 3
output_weights = (np.random.random((10, 1024)) - 0.5) * 2 / 3
output_bias = (np.random.random((10, 1)) - 0.5) * 2 / 3


def train():
    print('Try reading the params...')
    if try_read_params():
        print('Using old parameters')
    loader = data_loader.MnistLoader()
    print('Reading the data set...')
    training_set, training_labels = loader.get_training_set(True)
    print('Start training...')
    batch_size = 10
    size = len(training_labels)
    # count = 40
    count = 20
    # alpha = 0.06
    alpha = 0.03
    pool = Pool(2)
    for i in range(count):
        start = time.time()
        print(i)
        result = pool.apply_async(train_function, (training_set, training_labels, size, batch_size,
                                                   kernel1, bias1, kernel2, bias2, fc_weights, fc_bias, output_weights, output_bias))
        result2 = pool.apply_async(train_function, (training_set, training_labels, size, batch_size,
                                                    kernel1, bias1, kernel2, bias2, fc_weights, fc_bias, output_weights, output_bias))
        gradients = train_function(training_set, training_labels, size, batch_size,
                                   kernel1, bias1, kernel2, bias2, fc_weights, fc_bias, output_weights, output_bias)
        # print(gradients[5])
        # print(gradients[6])
        update_params(alpha, gradients)
        # r = result.get()
        # print(r[6])
        update_params(alpha, result.get())
        update_params(alpha, result2.get())
        print((time.time() - start) / 60)
        if i % 50 == 0:
            alpha -= 0.005
        if i % 10 == 0:
            save()
    pool.close()
    pool.join()
    print('end')
    save()


def train_function(training_set, training_labels, size, batch_size, k1, b1, k2, b2, fw, fb, ow, ob):
    images = []
    labels = []
    np.random.seed()
    indeces = np.random.randint(0, size, batch_size)
    # print(indeces)
    for j in indeces:
        images.append(training_set[j])
        labels.append(training_labels[j])
    return bp(images, labels, k1, b1, k2, b2, fw, fb, ow, ob)


def test():
    loader = data_loader.MnistLoader()
    test_set, test_labels = loader.get_test_set()
    count = len(test_set)
    right_count = 0
    for i in range(count):
        prediction = predict(test_set[i], test_labels[i])
        if prediction == test_labels[i]:
            right_count += 1
    print('The accuracy is %f' % (right_count / count))


def bp(images, labels, k1, b1, k2, b2, fw, fb, ow, ob):
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
        conv1_result = convolution(image, k1, b1)
        conv1_act_result.append(ReLU(conv1_result))
        pool1_result.append(max_pool(conv1_act_result[i]))
        conv2_result = convolution(pool1_result[i], k2, b2)
        conv2_act_result.append(ReLU(conv2_result))
        pool2_result.append(max_pool(conv2_act_result[i]))
        vector = pool2_result[i].reshape(7 * 7 * 64, 1)
        fc_result = np.matmul(fw, vector) + fb  # 1024 * 1
        fc_act_result.append(ReLU(fc_result, True))
        raw_output = np.matmul(ow, fc_act_result[i]) + ob
        output.append(softmax(raw_output))
    output_error = np.zeros(10)
    for j in range(count):
        for i in range(10):
            output_error[i] += ((output[j][i]) + (-1 if i == labels[j] else 0)) * (1 / count)
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
        fc_error = np.matmul(ow.T, output_error) * get_derivative(fc_act_result[i])  # 1024 * 1
        vector = pool2_result[i].reshape(7 * 7 * 64, 1)
        fc_weight_gradient.append(np.matmul(fc_error, vector.T))
        fc_bias_gradient.append(fc_error)
        pool2_error = np.matmul(fw.T, fc_error)  # (7 * 7 * 64) * 1
        conv2_error = get_conv_error(conv2_act_result[i], pool2_result[i], pool2_error) * get_derivative(
            conv2_act_result[i])  # 64 * 14 * 14
        kernel2_gradient.append(get_kernel_derivative(conv2_error, pool1_result[i]))  # 64 * 32 * 3 * 3
        bias2_gradient.append(conv2_error.reshape(14 * 14 * 64))
        pool1_error = back_conv(conv2_error, flip(k2))  # 32 * 14 * 14
        conv1_error = get_conv_error(conv1_act_result[i], pool1_result[i], pool1_error) * get_derivative(
            conv1_act_result[i])  # 32 * 28 * 28
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
    return output_weights_g, output_bias_g, fc_weight_g, fc_bias_g, kernel2_g, bias2_g, kernel1_g, bias1_g


def update_params(alpha, gradients):
    global output_weights
    global output_bias
    global fc_weights
    global fc_bias
    global kernel2
    global bias2
    global kernel1
    global bias1
    output_weights -= alpha * gradients[0]
    output_bias -= alpha * gradients[1]
    fc_weights -= alpha * gradients[2]
    fc_bias -= alpha * gradients[3]
    kernel2 -= alpha * gradients[4]
    bias2 -= alpha * gradients[5]
    kernel1 -= alpha * gradients[6]
    bias1 -= alpha * gradients[7]


def predict(image, label):
    image = image
    conv1_result = convolution(image, kernel1, bias1)
    conv1_act_result = ReLU(conv1_result)
    pool1_result = max_pool(conv1_act_result)
    conv2_result = convolution(pool1_result, kernel2, bias2)
    conv2_act_result = ReLU(conv2_result)
    pool2_result = max_pool(conv2_act_result)
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
    print('The probability of \'%d\' is %f. And the true label is %d' % (index, p, label))
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
                        result[k][i * 2 + x][j * 2 + y] = e[k][i][j] if pool_result[k][i][j] == \
                                                                        conv_result[k][i * 2 + x][j * 2 + y] else 0
    return result


def flip(kernels):
    return np.flip(np.flip(kernels, -1), -2)


def convolution(data, kernels, bias=None, flipped=False):
    data_shape = np.shape(data)
    if len(data_shape) < 3:
        data = data.reshape(1, data_shape[0], data_shape[1])
        data_shape = np.shape(data)
    data_width = data_shape[2]
    data_height = data_shape[1]
    kernel_shape = np.shape(kernels)
    kernel_count = kernel_shape[0]
    kernel_len = kernel_shape[1]
    conv_image = np.empty((kernel_count, data_height, data_width))
    for i in range(kernel_count):
        conv = np.zeros((data_height, data_width))
        for j in range(kernel_len):
            if flipped:
                conv += signal.convolve2d(data[j], kernels[i][j], mode='same')
            else:
                conv += signal.correlate2d(data[j], kernels[i][j], mode='same')
        conv_image[i] = conv
    b = 0
    if bias is not None:
        b = bias.reshape((kernel_count, data_height, data_width))
    return conv_image + b


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
    return convolution(error, kernel, flipped=True)


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
                    result[i][j][x + 1][y + 1] = signal.convolve2d(error[j], kernel, 'valid')
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
    count = len(result)
    maxima = max(result)
    result -= maxima
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


if __name__ == '__main__':
    train()
    # test()
    # try_read_params()
    # print(kernel1)
