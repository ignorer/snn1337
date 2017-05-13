%%cython
cimport numpy as np
from neuron import *

class InputNeuron(object):
    
    def __init__(self, nnet, spike_train):
        self.spike_train = spike_train
        self.output_spikes_times = []
        self.net = nnet
        
    def set_spike_train(self, spike_train):
        self.spike_train = spike_train
        self.output_spikes_times = []
        
    def step(self):
        if self.spike_train[self.net.global_time] == 1:
            self.output_spikes_times.append(self.net.global_time)
    
    def get_spikes(self):
        return self.output_spikes_times


class Connection(object):
    def __init__(self, nnet, input_neuron, output_neuron,
                 weights=[1], delays=[1]):  # weights and delays are scaled
        self.weights = weights
        self.delays = delays
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.net = nnet
        self.last_conducted_spike_index = 0

    def step(self):
        spikes = self.input_neuron.output_spikes_times
        for i in range(self.last_conducted_spike_index, len(spikes)):
            spike_time = spikes[i]
            for j in range(len(self.weights)):
                if spike_time + self.delays[j] == self.net.global_time:
                    self.last_conducted_spike_index += 1
                    self.output_neuron.receive_spike(self.weights[j])
