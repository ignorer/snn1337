transcript on

vlib work

vlog -sv +incdir+./ ./random_lfsr.sv
vlog -sv +incdir+./ ./random_lfsr_tb.sv

vsim -t 1ns -voptargs="+acc" random_lfsr_tb

run -all

