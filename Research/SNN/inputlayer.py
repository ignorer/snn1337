%%cython

cimport numpy as np
from neuron import *

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
def fixed_frequency_spike_train(frequency, t_max):
    actual_frequency = float(frequency)
    result = [1 if frequency > 0 else 0]
    for i in range(t_max - 1):
        if actual_frequency >= 1:
            result.append(1)
            actual_frequency -= int(actual_frequency)
        else:
            result.append(0)
        actual_frequency += frequency
    return result

print(fixed_frequency_spike_train)

from functools import reduce
class InputLayer(object):
    def __init__(self, nnet, shape):
        self.net = nnet
        self.shape = shape 
        self.neur_size = reduce(lambda res, x: res*x, self.shape, 1)
        self.neurons = np.ndarray(shape=self.shape, dtype=InputNeuron, buffer=np.array([InputNeuron(self.net, []) for i in np.arange(self.neur_size)]))
            
    def random_spike_train(freq, t_max):
        return sps.bernoulli.rvs(0.25*freq +0.5, size=t_max)
    
    def new_input(self, arg, t_max=1000, make_spike_train=fixed_frequency_spike_train):
        for i, f in enumerate(arg):
            for j, l in enumerate(f):
                for k, m in enumerate(l):
                    self.neurons[i][j][k].set_spike_train(fixed_frequency_spike_train(arg[i][j][k], t_max))
    
    def step(self):
        for neur in self.neurons.reshape((self.neur_size)):
            neur.step()
