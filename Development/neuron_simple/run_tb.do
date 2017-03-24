transcript on

vlib work

vlog -sv +incdir+./ ./neuron_simple_example.sv
vlog -sv +incdir+./ ./neural_network_simple_example.sv
vlog -sv +incdir+./ ./example_tb.sv

vsim example_tb

run -all
