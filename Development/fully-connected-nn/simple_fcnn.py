import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1. - x)

def fpga_sigmoid_simple(x):
    abs_x = np.abs(x)
    if (abs_x >= 5):
        y = 1.
    elif (abs_x >= 2.375):
        y = 0.03125 * abs_x + 0.84375
    elif (abs_x >= 1):
        y = 0.125 * abs_x + 0.625
    else:
        y = 0.25 * abs_x + 0.5
    return y

def fpga_sigmoid(x):
    return np.vectorize(fpga_sigmoid_simple)(x)

class FCNN():
    def __init__(self, layer_sizes, seed = 1337, activation = sigmoid, activation_derivative = sigmoid_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
        np.random.seed(seed)
        self.weights = []
        for i in range(1, len(layer_sizes)):
            self.weights.append(2. * np.random.random((layer_sizes[i - 1], layer_sizes[i])) - 1.)

    def predict(self, input):
        layer_res = [self.activation(np.dot(input, self.weights[0]))]
        for i in range(1, len(self.weights)):
            layer_res.append(self.activation(np.dot(layer_res[i - 1], self.weights[i])))
        return layer_res

    def fit(self, inputs, outputs, rate, iterations):
        for i in range(0, iterations):
            layer_res = self.predict(inputs)
            prev_delta = (layer_res[-1] - outputs) * self.activation_derivative(layer_res[-1])

            for i in range(len(self.weights) - 2, -1, -1):
                next_delta = np.dot(prev_delta, self.weights[i + 1].T) * self.activation_derivative(layer_res[i])
                self.weights[i + 1] -= rate * np.dot(layer_res[i].T, prev_delta)
                prev_delta = next_delta

            self.weights[0] -= rate * np.dot(inputs.T, prev_delta)

    def get_fpga_network(self, decimal_precision):
        res = []
        for layer in self.weights:
            res.append((layer * (10 ** decimal_precision) + 0.5).astype(int))
        return res

if __name__ == "__main__":
    net = FCNN([2, 2, 1])

    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_outputs = np.array([[0, 1, 1, 0]]).T
    net.fit(training_inputs, training_outputs, 1, 10000)

    print(net.predict(np.array([0, 0]))[-1])
    print(net.predict(np.array([0, 1]))[-1])
    print(net.predict(np.array([1, 0]))[-1])
    print(net.predict(np.array([1, 1]))[-1])

