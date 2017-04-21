transcript on

vlib work

vlog -sv +incdir+./ ./spiking_neuron.sv
vlog -sv +incdir+./ ./spiking_neural_network_xor.sv
vlog -sv +incdir+./ ./spiking_example_network_tb.sv

vsim -t 1ns -voptargs="+acc" example_network_tb
add wave /example_network_tb/clk
add wave /example_network_tb/rst
add wave /example_network_tb/neural_network/in1
add wave /example_network_tb/neural_network/in2
add wave -radix decimal /example_network_tb/neural_network/counter
add wave -radix decimal /cmd
add wave /example_network_tb/neural_network/neuron_0_1_out
add wave /example_network_tb/neural_network/neuron_0_2_out
add wave /example_network_tb/neural_network/neuron_1_1_out
add wave /example_network_tb/neural_network/neuron_1_2_out
add wave /example_network_tb/neural_network/neuron_2_1_out

configure wave -timelineunits ns
run -all
wave zoom full

