from data import data_loader
import math
import random


class Neuron:
    def __init__(self, is_random=True, bias=0, lower=-1, upper=1):
        if is_random:
            self.b = random.uniform(lower, upper)
        else:
            self.b = bias

    def output(self, input_data):
        pass


class ReLU(Neuron):
    def output(self, input_data):
        return max(0, input_data)


class SigmoidUnit(Neuron):
    def output(self, input_data):
        return 1 / (1 + math.pow(math.e, -input_data))


class Layer:
    ACT_FUNC_ReLU = 0
    ACT_FUNC_SigmoidUnit = 1

    def __init__(self, count, last_layer, next_layer, activation_function=ACT_FUNC_ReLU):
        self.count = count
        self.units = []
        self.last_layer = last_layer
        self.next_layer = next_layer
        self._init_units(activation_function)

    def _init_units(self, activation_function):
        if activation_function == Layer.ACT_FUNC_ReLU:
            for i in range(self.count):
                self.units.append(ReLU())
        elif activation_function == Layer.ACT_FUNC_SigmoidUnit:
            for i in range(self.count):
                self.units.append(SigmoidUnit)
        else:
            raise Exception('Wrong type of activation unit')


class ConvLayer(Layer):
    pass
