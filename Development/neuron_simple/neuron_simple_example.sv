/*
Module neuron_example_simple

It is a simple model of neuron
Inputs and outputs:
  clk, rst - service inputs
  in1, in2 - neural inputs
  out - neural output
in1, in2, out - integers with INT_WIDTH bits 
(integer range [0..2^INT_WIDTH) is mapped to real [0, 1))

Required parameters:
  INT_WIDTH - size of input integers in bits, and width of base of weight parameters
    so if INT_WIDTH = 4, then
      1) "in1 = 15" - correct and 
         "in1 = 16" - not correct,
         "in1 = 8" means 0.5,
         "in1 = 4" means 0.25,
         "in1 = 0" means 0.0.
      2) "IN1_WEIGHT = 16" means 1.0, 
         "IN1_WEIGHT = 32" means 2.0, 
         "IN1_WEIGHT = 8" means 0.5.
  NEURON_LEVEL, NEURON_ID - coordinates of neuron (used mainly for debug)
  OVERFLOW_WIDTH - Upper rounding of log_2 of arguments count, 
    defines count of additional bits for integer overflow.
  IN1_WEIGHT, IN2_WEIGHT - weights.
  
Note: neuron does not use barrier function (ot it's primitive x -> min(0, max(x, INT_MAX)))
*/

module neuron_example_simple(clk, rst, in1, in2, out);
  
  parameter NEURON_LEVEL;
  parameter NEURON_ID; // id inside level
  
  parameter INT_WIDTH;
  parameter INT_MSB = INT_WIDTH - 1;
  parameter INT_MAX = (1 << INT_WIDTH) - 1;
  
  parameter OVERFLOW_WIDTH = 2; // upper rounding of log2 of arguments count
  
  parameter IN1_WEIGHT;
  parameter IN2_WEIGHT;
    
  //initial $display("Neuron(%d, %d) initialised, weights: [%d, %d]", 
  //    NEURON_LEVEL, NEURON_ID, IN1_WEIGHT, IN2_WEIGHT);
  
  input wire clk;
  input wire rst;
  
  input wire [INT_MSB:0] in1;
  input wire [INT_MSB:0] in2;
  
  output wire [INT_MSB:0] out;
  
  parameter SIGN_WIDTH = 1;
  parameter SUM_WIDTH = INT_WIDTH * 2 + OVERFLOW_WIDTH + SIGN_WIDTH;
  parameter SUM_MSB = SUM_WIDTH - 1;
  
  parameter SIGN_MSB = SUM_MSB;
  parameter SIGN_LSB = SUM_MSB - SIGN_WIDTH + 1;
  parameter OVERFLOW_MSB = SIGN_LSB - 1;
  parameter OVERFLOW_LSB = OVERFLOW_MSB - OVERFLOW_WIDTH + 1;
  parameter SUM_NEEDED_MSB = OVERFLOW_LSB - 1;
  parameter SUM_NEEDED_LSB = SUM_NEEDED_MSB - INT_WIDTH + 1;
  
  
  //initial $display("SIGN[%d:%d] OVERFLOW[%d:%d] NEEDED[%d:%d]", 
  //    SIGN_MSB, SIGN_LSB, OVERFLOW_MSB, OVERFLOW_LSB, SUM_NEEDED_MSB, SUM_NEEDED_LSB);
    
  wire [SUM_MSB:0] sum;
  
  // note:
  // [                             sum                                 ]
  // [sign      ] [overflow      ] [sum_needed] [not used ]
  //  SIGN_WIDTH   OVERFLOW_WIDTH   INT_WIDTH    INT_WIDTH 
  assign sum = in1 * IN1_WEIGHT + in2 * IN2_WEIGHT;
  
  wire is_negative;
  wire is_overflow;
  wire [INT_MSB:0] needed;
  
  assign is_negative = (sum[SIGN_MSB:SIGN_LSB] != 0);
  assign is_overflow = (sum[OVERFLOW_MSB:OVERFLOW_LSB] != 0);
  assign needed = sum[SUM_NEEDED_MSB:SUM_NEEDED_LSB];
  
  assign out = (is_negative ? 0 : (is_overflow ? INT_MAX : needed));
  
  // Relatively useless debug element
  always @(posedge clk)
  begin
    $display("Sum = %d, IsNeg = %d, IsOverflow = %d, Needed = %d", 
        sum, is_negative, is_overflow, needed);  
  end
  
endmodule