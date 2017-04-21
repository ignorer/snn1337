/*
Module spiking_neural_network_xor

It is a simple module that counts XOR function
Uses neuron_example_simple inside
*/

module spiking_neural_network_xor(clk, rst, addr, cmd, cmd_arg, in1, in2, out);
  parameter INT_WIDTH = 4;
  parameter INT_MSB = INT_WIDTH - 1;
  parameter INT_MAX = (1 << INT_WIDTH) - 1;
  parameter WEIGHT_ONE = (1 << INT_WIDTH);
  
  parameter FLOAT_WIDTH = INT_WIDTH * 2;
  
  parameter CMD_WIDTH = 3;
  parameter ADDR_WIDTH = 3;
  
  parameter SILENT = 1;
  
  input wire clk;
  input wire rst;
  wire neurons_rst;
  
  input wire in1;
  input wire in2;
  
  output reg out;

  input reg [CMD_WIDTH - 1 : 0] cmd;
  input reg [ADDR_WIDTH - 1 : 0] addr;
  input reg [FLOAT_WIDTH - 1 : 0] cmd_arg;
  
  reg[31:0] counter;
  reg preout;
  
  reg neuron_0_1_out;
  reg neuron_0_2_out;
    
  wire neuron_1_1_out;
  wire neuron_1_2_out;
  wire neuron_2_1_out;
    
  spiking_neuron_2in #(
    .NEURON_LEVEL(1), .NEURON_ID(1), .NEURON_GLOBAL_ID(1), 
    .INT_WIDTH(INT_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .CMD_WIDTH(CMD_WIDTH)
  ) neuron_1_1(.clk(clk), .rst(rst), .addr(addr), .cmd(cmd), .cmd_arg(cmd_arg), 
      .in1(neuron_0_1_out), .in2(neuron_0_2_out), .out(neuron_1_1_out));
  
  spiking_neuron_2in #(
    .NEURON_LEVEL(1), .NEURON_ID(2), .NEURON_GLOBAL_ID(2), 
    .INT_WIDTH(INT_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .CMD_WIDTH(CMD_WIDTH)
  ) neuron_1_2(.clk(clk), .rst(rst), .addr(addr), .cmd(cmd), .cmd_arg(cmd_arg), 
      .in1(neuron_0_1_out), .in2(neuron_0_2_out), .out(neuron_1_2_out));
  
  spiking_neuron_2in #(
    .NEURON_LEVEL(2), .NEURON_ID(1), .NEURON_GLOBAL_ID(3), 
    .INT_WIDTH(INT_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .CMD_WIDTH(CMD_WIDTH), .SILENT(0)
  ) neuron_2_1(.clk(clk), .rst(rst), .addr(addr), .cmd(cmd), .cmd_arg(cmd_arg),
      .in1(neuron_1_1_out), .in2(neuron_1_2_out), .out(neuron_2_1_out));
  
  always @(posedge rst)
  begin
    if (!SILENT) $display("Neural network started");
    counter = -1;
    out = 'b z;
    neuron_0_1_out = 0;
    neuron_0_2_out = 0;
    preout = 0;
  end
  
  always @(posedge clk)
  begin
    if (!SILENT) $display("Neural network: New clk! (%3d)", counter);
    case (counter)
      10 : begin
        if (!SILENT) $display("Neural network: Start argument pulses");
        neuron_0_1_out = in1;
        neuron_0_2_out = in2;
      end
      11 : begin
        if (!SILENT) $display("Neural network: Finish argument pulses");
        neuron_0_1_out = 0;
        neuron_0_2_out = 0;
      end
      30 : begin
        out = preout;
      end   
    endcase;
    if (neuron_2_1_out)
      preout = 1;
    counter = counter + 1;    
  end
  
  
endmodule
