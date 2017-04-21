transcript on

vlib work

vlog -sv +incdir+./ ./spiking_neuron.sv
#vlog -sv +incdir+./ ./neural_network_simple_example.sv
vlog -sv +incdir+./ ./spiking_example_tb.sv

#vsim example_tb

vsim -t 1ns -voptargs="+acc" example_tb
add wave /example_tb/clk
add wave /example_tb/rst
add wave /example_tb/neuron1_in1
add wave /example_tb/neuron1_in2
add wave /example_tb/neuron1_out


add wave -radix decimal /example_tb/addr
add wave -radix decimal /example_tb/cmd
add wave -radix decimal /example_tb/weight

add wave -radix decimal /example_tb/neuron1/state1
add wave -radix decimal /example_tb/neuron1/state2
add wave -radix decimal /example_tb/neuron1/state_out
configure wave -timelineunits ns
run -all
wave zoom full

