transcript on

vlib work

vlog -sv +incdir+./ ./random_lfsr.sv
vlog -sv +incdir+./ ./spiking_neuron.sv
vlog -sv +incdir+./ ./spiking_neural_network_xor.sv
vlog -sv +incdir+./ ./anneal_spike_tb.sv

vsim -t 1ns -voptargs="+acc" anneal_spike_tb
add wave /anneal_spike_tb/clk
add wave /anneal_spike_tb/rst
add wave /anneal_spike_tb/neural_network/in
add wave -radix decimal /anneal_spike_tb/neural_network/counter
add wave -radix decimal /cmd
add wave /anneal_spike_tb/neural_network/neuron_1_out
add wave /anneal_spike_tb/neural_network/neuron_2_out
add wave /anneal_spike_tb/neural_network/neuron_3_out
add wave /anneal_spike_tb/neural_network/neuron_4_out
add wave /anneal_spike_tb/neural_network/neuron_5_out
add wave /anneal_spike_tb/neural_network/neuron_6_out
add wave /anneal_spike_tb/neural_network/neuron_7_out

configure wave -timelineunits ns
run -all
wave zoom full


