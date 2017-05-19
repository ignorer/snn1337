#! /usr/bin/python2

from generate_net import *

net = SnnGenerator(
            net_module_name="net_sample_1",
            destination_file="net_sample_1.sv",
            template_file="spiking_neural_network.sv.template",
            neuron_template_file="spiking_neuron.sv.template")

net.add_input_layer(2)

net.add_dense_layer(4)

net.add_dense_layer(3)

net.add_output_level(2)

net.generate_net_code()
