/*
Module spiking_neural_network_xor

It is a simple module that counts XOR function
Uses neuron_example_simple inside
*/

module spiking_neural_network_xor(clk, rst, in1, in2, out);
  parameter INT_WIDTH;
  parameter INT_MSB = INT_WIDTH - 1;
  parameter INT_MAX = (1 << INT_WIDTH) - 1;
  parameter WEIGHT_ONE = (1 << INT_WIDTH);
  
  parameter SILENT = 1;
  
  input wire clk;
  input wire rst;
  wire neurons_rst;
  
  input wire in1;
  input wire in2;
  
  output reg out;

  reg[31:0] counter;
  reg preout;
  
  reg neuron_0_1_out;
  reg neuron_0_2_out;
    
  wire neuron_1_1_out;
  wire neuron_1_2_out;
  wire neuron_2_1_out;
    
  spiking_neuron_2in #(
    .NEURON_LEVEL(1), .NEURON_ID(1), .INT_WIDTH(INT_WIDTH), 
    .IN1_WEIGHT(WEIGHT_ONE / 2.1), .IN2_WEIGHT(WEIGHT_ONE / 2.1)
  ) neuron_1_1(.clk(clk), .rst(rst), .in1(neuron_0_1_out), .in2(neuron_0_2_out), .out(neuron_1_1_out));
  
  spiking_neuron_2in #(
    .NEURON_LEVEL(1), .NEURON_ID(2), .INT_WIDTH(INT_WIDTH), 
    .IN1_WEIGHT(WEIGHT_ONE * 1.2), .IN2_WEIGHT(WEIGHT_ONE * 1.2)
  ) neuron_1_2(.clk(clk), .rst(rst), .in1(neuron_0_1_out), .in2(neuron_0_2_out), .out(neuron_1_2_out));
  
  spiking_neuron_2in #(
    .NEURON_LEVEL(2), .NEURON_ID(1), .INT_WIDTH(INT_WIDTH), 
    .IN1_WEIGHT(-WEIGHT_ONE), .IN2_WEIGHT(WEIGHT_ONE)
  ) neuron_2_1(.clk(clk), .rst(rst), .in1(neuron_1_1_out), .in2(neuron_1_2_out), .out(neuron_2_1_out));
  
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
      0 : begin
        if (!SILENT) $display("Neural network: Start argument pulses");
        neuron_0_1_out = in1;
        neuron_0_2_out = in2;
      end
      1 : begin
        if (!SILENT) $display("Neural network: Finish argument pulses");
        neuron_0_1_out = 0;
        neuron_0_2_out = 0;
      end
      20 : begin
        out = preout;
      end   
    endcase;
    if (neuron_2_1_out)
      preout = 1;
    counter = counter + 1;    
  end
  
  
endmodule
