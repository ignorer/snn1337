import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1. - x)


def fpga_sigmoid_simple(x):
    abs_x = np.abs(x)
    if abs_x >= 5:
        y = 1.
    elif abs_x >= 2.375:
        y = 0.03125 * abs_x + 0.84375
    elif abs_x >= 1:
        y = 0.125 * abs_x + 0.625
    else:
        y = 0.25 * abs_x + 0.5
    if x < 0:
        y = 1. - y
    return y

def fpga_sigmoid_derivative_simple(x):
    abs_x = np.abs(x)
    if abs_x >= 5:
        y = 0
    elif abs_x >= 2.375:
        y = 0.03125
    elif abs_x >= 1:
        y = 0.125
    else:
        y = 0.25
    return y


def fpga_sigmoid(x):
    return np.vectorize(fpga_sigmoid_simple)(x)

def fpga_sigmoid_derivative(x):
    return np.vectorize(fpga_sigmoid_derivative_simple)(x)

def get_random_network(layer_sizes, seed=1337, activation=fpga_sigmoid, activation_derivative=fpga_sigmoid_derivative):
    np.random.seed(seed)
    weights = []
    bias = []
    for i in range(1, len(layer_sizes)):
        bias.append(2. * np.random.random(layer_sizes[i]) - 1.)
        weights.append(2. * np.random.random((layer_sizes[i - 1], layer_sizes[i])) - 1.)
    return FCNN(weights, bias, activation, activation_derivative)

class FCNN:
    def __init__(self, weights, bias, activation=fpga_sigmoid, activation_derivative=fpga_sigmoid_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.weights = weights
        self.bias = bias

    def predict(self, input):
        layer_res = [self.activation(np.dot(input, self.weights[0]) + self.bias[0])]
        for i in range(1, len(self.weights)):
            layer_res.append(self.activation(np.dot(layer_res[i - 1], self.weights[i]) + self.bias[i]))
        return layer_res

    def fit(self, inputs, outputs, rate, iterations):
        for i in range(0, iterations):
            layer_res = self.predict(inputs)
            prev_delta = (layer_res[-1] - outputs) * self.activation_derivative(layer_res[-1])
            for i in range(len(self.weights) - 2, -1, -1):
                next_delta = np.dot(prev_delta, self.weights[i + 1].T) * self.activation_derivative(layer_res[i])
                self.weights[i + 1] -= rate * np.dot(layer_res[i].T, prev_delta)
                self.bias[i + 1] -= rate * np.sum(prev_delta, axis=0)
                prev_delta = next_delta

            self.weights[0] -= rate * np.dot(inputs.T, prev_delta)
            self.bias[0] -= rate * np.sum(prev_delta, axis=0)

    def get_layer_sizes(self):
        return [self.weights[0].shape[0]] + [self.weights[i].shape[1] for i in range(len(self.weights))]

    def get_fpga_weights(self, decimal_precision):
        res = []
        for layer in self.weights:
            res.append((layer * (10 ** decimal_precision) + 0.5).astype(int))
        return res

    def get_fpga_bias(self, decimal_precision):
        res = []
        for layer in self.bias:
            res.append((layer * (10 ** decimal_precision) + 0.5).astype(int))
        return res


if __name__ == "__main__":
    net = get_random_network([2, 2, 1])

    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_outputs = np.array([[0, 1, 1, 0]]).T
    net.fit(training_inputs, training_outputs, 1, 10000)

    #print(fpga_sigmoid_simple(net.bias[0]))

    print(net.predict(np.array([0, 0]))[-1])
    print(net.predict(np.array([0, 1]))[-1])
    print(net.predict(np.array([1, 0]))[-1])
    print(net.predict(np.array([1, 1]))[-1])
