%%cython

cimport numpy as np
from neuron import *
from base import *
from inputlayer import *
from convlayer import *
from denselayer import *

class NNet(object):
    def __init__(self, shape, threshold=1.):
        self.layers = [InputLayer(self, shape)]
        self.global_time = 0
        self.threshold = threshold
    
    def add_convolution(self, weights):
        num_filters = weights.shape[0]
        filter_shape = weights.shape[2:4]
        self.layers.append(Conv2DLayer(self, self.layers[-1], num_filters, filter_shape, weights))
        
    def add_subsampling(self, pool_size):
        self.layers.append(SubSampling2DLayer(self, self.layers[-1], pool_size))
        
    def add_dense(self, weights):
        num_units = weights.shape[1]
        self.layers.append(DenseLayer(self, self.layers[-1], num_units, weights, threshold=self.threshold))
    
    def get_output_for(self, data, t_max):
        self.global_time = 0
        self.layers[0].new_input(data, t_max)
        for l in self.layers[1:]:
            l.restart()
        for t in np.arange(t_max):
            #for layer in self.layers:
                #layer.step()
            list(map(lambda x: x.step(), self.layers)) # POOL
            self.global_time += 1
        result = [neur.get_spikes() for neur in self.layers[-1].neurons.reshape((self.layers[-1].neur_size))]
        return result
    
    def classify(self, data, t_max):
        self.global_time = 0
        self.layers[0].new_input(data)
        for l in self.layers[1:]:
            l.restart()
        ans = []
        for t in np.arange(t_max):
            #for layer in self.layers:
                #layer.step()
            list(map(lambda x: x.step(), self.layers)) # POOL
            for i, neur in enumerate(self.layers[-1].neurons):
                if len(neur.get_spikes()) > 0:
                    ans.append(i)
            if(len(ans) > 0):
                return ans, t
            self.global_time += 1
        print('not_enough_time')

def spiking_from_lasagne(input_net, threshold):
    import lasagne
    input_layers = lasagne.layers.get_all_layers(input_net)
    weights = lasagne.layers.get_all_param_values(input_net)
    spiking_net = NNet(input_layers[0].shape[-3:], threshold)
    convert_layers = {lasagne.layers.conv.Conv2DLayer : spiking_net.add_convolution,\
                      lasagne.layers.dense.DenseLayer : spiking_net.add_dense}
    
    #номер элемента в общем массиве весов, в котором хранятся веса текущего слоя
    i = 0
    
    for l in input_layers[1:]:
        if(type(l) == lasagne.layers.pool.Pool2DLayer or type(l) == lasagne.layers.pool.MaxPool2DLayer):
            spiking_net.add_subsampling(l.pool_size)
        else:
            convert_layers[type(l)](weights[i])
            i+=1
            
    return spiking_net