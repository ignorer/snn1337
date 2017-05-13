# cython: cdivision=True
# cython: profile=True
from libcpp.vector cimport vector
from libc.math cimport exp, fabs
from libcpp.utility cimport pair

ctypedef pair[int,double] pairID


cdef class Neuron(object):
    
    cdef double potential, tau_m, tau_s, tau_r, time_scale, threshold
    cdef int last_output_spikes_time, global_time
    cdef vector[int] output_spikes_times
    cdef vector[pairID] input_spikes
    cdef vector[double] history
    
    cdef _cinit__(self, double threshold):
        self.potential = 0
        self.threshold = threshold
        self.tau_m = 4
        self.tau_s = 2
        self.tau_r = 20
        self.time_scale = 1 # time unit is 100 ms
        self.last_output_spikes_time = 0
        global_time = 0
    
    cdef void _receive_spike(self, double intensity, int global_time):
        self.input_spikes.push_back(pair[int,double](global_time, intensity))
    
    cdef void _restart(self):
        self.potential = 0
        self.output_spikes_times.clear()
        self.input_spikes.clear()
        self.history.clear()
        self.last_output_spikes_time = 0
        global_time = 0
    
    cdef void _step(self):
        cdef int global_time = self.global_time
        self.potential = 0
        cdef int spike_time
        cdef double intensity
            
        for i in range(self.input_spikes.size()):
            spike_time = self.input_spikes[i].first
            intensity = self.input_spikes[i].second
            if self.time_scale * (global_time - spike_time) < 30 and (fabs(intensity) > 0.0001):
                self.potential += self._eps(global_time - spike_time) * intensity

        self.potential += self._nu(global_time - self.last_output_spikes_time)
        self.history.push_back(self.potential)

        if self.potential > self.threshold:
            self.input_spikes.clear()
            self.output_spikes_times.push_back(global_time)
            self.last_output_spikes_time = global_time
        global_time += 1
    
    cdef double _eps(self, int time):
        if time <= 0:
            return 0
        cdef double s = -fabs(time * self.time_scale)
        cdef double result = exp(s / self.tau_m) - exp(s / self.tau_s)
        return result

    cdef double _nu(self, int time):
        if time <= 0:
            return 0
        cdef double s = time * self.time_scale
        cdef double result = -self.threshold * exp(-fabs(s) / self.tau_r) * (s > 0)
        return result

    cdef vector[int] _get_spikes(self):
        return self.output_spikes_times

    cdef vector[double] _get_history(self):
        return self.history

    def __init__(self, nnet, threshold=1.):
        self._cinit__(threshold)
        
    def restart(self):
        self._restart()
    
    def receive_spike(self, intensity):
        return self._receive_spike(intensity, self.global_time)
        
    def step(self):
        self._step()
        
    def get_spikes(self):
        return self._get_spikes()
    def get_history(self):
        return self._get_history()

    

# cdef class _Connection(object):
#     cdef int last_conducted_spike_index
#     cdef vector[double] weights
#     cdef vector[int] delays
#     cdef Neuron output_neuron
    
#     def __init__(self, output_neuron, weights=[1], delays=[1]):  # weights and delays are scaled
#         self.weights = weights
#         self.delays = delays
#         self.output_neuron = output_neuron
#         self.last_conducted_spike_index = 0

#     def step(self, spikes, global_time):
#         self._step(spikes, global_time)
    
#     cdef void _step(self, vector[int]& spikes, int global_time):
#         cdef int spike_time
#         for i in range(self.last_conducted_spike_index, spikes.size()):
#             spike_time = spikes[i]
#             for j in range(self.weights.size()):
#                 if spike_time + self.delays[j] == global_time:
#                     self.last_conducted_spike_index += 1
#                     self.output_neuron._receive_spike(self.weights[j], global_time)

# class Connection(object):
    
#     def __init__(self, nnet, input_neuron, output_neuron,
#                  weights=[1], delays=[1]):  # weights and delays are scaled
#         self.cConnection = _Connection(output_neuron, weights, delays)
#         self.input_neuron = input_neuron
#         self.net = nnet

#     def step(self):
#         spikes = self.input_neuron.get_spikes()
#         self.cConnection.step(spikes, self.net.global_time)