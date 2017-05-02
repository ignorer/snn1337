import simple_fcnn
import testbench_generator
import numpy as np
from mnist import MNIST

with np.load('sigmoid_net_weights.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

weights = param_values[::2]
bias = param_values[1::2]

mndata = MNIST('datasets')

images, labels = mndata.load_training()
images, labels = mndata.load_testing()

net = simple_fcnn.FCNN(weights, bias)

generator = testbench_generator.FCNNTestbenchGenerator(net)
output = generator.generate_testbench([np.array(images[i]).astype(float) for i in range(len(images))])
with open('mnist_test.sv', 'w') as output_file:
    output_file.write(output)