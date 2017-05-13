%%cython

cimport numpy as np
from neuron import *
from base import *

class Conv2DLayer(object):
    # Формат весов: w[nk][h][i][j], где nk - фильтр, h - номер фильтра на предыдущем слое, i, j - координаты весов в фильтре
    def __init__(self, nnet, input_layer, num_filters, filter_shape, weights):
        self.net = nnet
        self.filter_shape = filter_shape
        self.weights = self.oldweights = weights.copy() 
        if(len(self.weights.shape) < 5):
            self.weights = self.weights.reshape(np.append(weights.shape, 1))
        
        self.shape = (num_filters, input_layer.shape[1]-filter_shape[0]+1, input_layer.shape[2]-filter_shape[1]+1)
        self.neur_size = reduce(lambda res, x: res*x, self.shape, 1)
        self.neurons = np.array([Neuron(self.net) for i in np.arange(self.neur_size)]).reshape(self.shape)
        
        self.connections = []
        
        for nk, kernel in enumerate(self.neurons):
            for i, row in enumerate(kernel):
                for j, neuron in enumerate(row):   # соединяем с предыдущим слоем
                    self.connections += [Connection(self.net, input_layer.neurons[l][i+p][j+q],neuron, self.weights[nk][l][p][q])\
                                         for l in np.arange(input_layer.shape[0]) for p in np.arange(filter_shape[0])\
                                         for q in np.arange(filter_shape[1])] 
                    
    def restart(self):
        for i, neur in enumerate(self.neurons.reshape((self.neur_size))):
            neur.restart()
    
    def step(self):
        for conn in self.connections:
            conn.step()
        for i, neur in enumerate(self.neurons.reshape((self.neur_size))):
            neur.step()