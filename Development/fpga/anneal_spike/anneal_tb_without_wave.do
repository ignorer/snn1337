transcript on

vlib work

vlog -sv +incdir+./ ./spiking_neuron.sv
vlog -sv +incdir+./ ./spiking_neural_network_xor.sv
vlog -sv +incdir+./ ./anneal_spike_tb.sv

vsim -t 1ns -voptargs="+acc" anneal_spike_tb
run -all

