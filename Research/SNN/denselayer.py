%%cython

cimport numpy as np
from neuron import *
from base import *

class DenseLayer(object):
    #Формат весов: w[i][j],  где i - номер нейрона на предыдущем слое, j - номер нейрона на текущем слое
    def __init__(self,nnet, input_layer, num_units, weights, threshold=1.):
        self.net = nnet
        self.shape = [num_units]
        self.neur_size = num_units
        self.neurons = np.array([Neuron(self.net, threshold) for i in np.arange(self.neur_size)])
        self.weights = weights
        
        if(len(weights.shape) < 3):
            weights = weights.reshape(np.append(weights.shape, 1))
        
        self.connections = [Connection(self.net, input_neuron, output_neuron, weights[i][j])\
                            for i, input_neuron in enumerate(input_layer.neurons.reshape((input_layer.neur_size)))\
                            for j, output_neuron in enumerate(self.neurons)]
        
    def restart(self):
        for neur in self.neurons:
            neur.restart()
        
    def step(self):
        #for conn in self.connections:
            #conn.step()
        list(map(lambda x: x.step(), self.connections)) # POOL
        list(map(lambda x: x.step(), self.neurons)) # POOL
        #for neur in self.neurons:
            #neur.step()