transcript on
vlib work

vlog -sv +incdir+./ ./sort5.sv
vlog -sv +incdir+./ ./sort.sv
vlog -sv +incdir+./ ./tb.sv

vsim tb

run -all

