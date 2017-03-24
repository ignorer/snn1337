/*
Module neural_network_simple

It is a simple module that counts XOR function
Uses neuron_example_simple inside
*/

module neural_network_simple(clk, rst, in1, in2, out);
  parameter INT_WIDTH;
  parameter INT_MSB = INT_WIDTH - 1;
  parameter INT_MAX = (1 << INT_WIDTH) - 1;
  parameter WEIGHT_ONE = (1 << INT_WIDTH);
  
  input wire clk;
  input wire rst;
  
  input wire in1;
  input wire in2;
  
  output wire out;

  
  wire [INT_MSB:0] neuron_1_1_out;
  wire [INT_MSB:0] neuron_1_2_out;
  
  wire [INT_MSB:0] in1_int;
  wire [INT_MSB:0] in2_int;
  wire [INT_MSB:0] out_int;
    
  neuron_example_simple #(
    .NEURON_LEVEL(1), .NEURON_ID(1), .INT_WIDTH(INT_WIDTH), 
    .IN1_WEIGHT(INT_MAX / 2), .IN2_WEIGHT(WEIGHT_ONE / 2)
  ) neuron_1_1(.clk(clk), .rst(rst), .in1(in1_int), .in2(in2_int), .out(neuron_1_1_out));
  
  neuron_example_simple #(
    .NEURON_LEVEL(1), .NEURON_ID(2), .INT_WIDTH(INT_WIDTH), 
    .IN1_WEIGHT(WEIGHT_ONE), .IN2_WEIGHT(WEIGHT_ONE)
  ) neuron_1_2(.clk(clk), .rst(rst), .in1(in1_int), .in2(in2_int), .out(neuron_1_2_out));
  
  neuron_example_simple #(
    .NEURON_LEVEL(2), .NEURON_ID(1), .INT_WIDTH(INT_WIDTH), 
    .IN1_WEIGHT(-WEIGHT_ONE), .IN2_WEIGHT(WEIGHT_ONE)
  ) neuron_2_1(.clk(clk), .rst(rst), .in1(neuron_1_1_out), .in2(neuron_1_2_out), .out(out_int));
  
  assign in1_int = in1 ? INT_MAX : 0;
  assign in2_int = in2 ? INT_MAX : 0;
  assign out = (out_int > INT_MAX / 2);
  
  // Relatively useless debug element
  always @(posedge clk)
  begin
    $display("in1i = %d, in2i = %d, out11 = %d, out12 = %d, outi = %d", 
        in1_int, in2_int, neuron_1_1_out, neuron_1_2_out, out_int);  
  end
  
endmodule
