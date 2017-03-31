/*
Module spiking_neuron_2in

It is a simple model of spiking neuron
Inputs and outputs:
  clk, rst - service inputs
  in1, in2 - neural inputs (wires)
  out - neural output (reg)
in1, in2, out - bits 
(integer range [0..2^INT_WIDTH) is mapped to real [0, 1))

Required parameters:
  INT_WIDTH - size of input integers in bits, and width of base of weight parameters
    so if INT_WIDTH = 4, then
      1) "in1 = 1" - correct and 
         "in1 = 0" - correct.
      2) "IN1_WEIGHT = 16" means 1.0, 
         "IN1_WEIGHT = 32" means 2.0, 
         "IN1_WEIGHT = 8" means 0.5, 
         "IN1_WEIGHT = -8" means 0.5.
  NEURON_LEVEL, NEURON_ID - coordinates of neuron (used mainly for debug)
  OVERFLOW_WIDTH - Upper rounding of log_2 of arguments count, 
    defines count of additional bits for integer overflow.
  IN1_WEIGHT, IN2_WEIGHT - weights.
  OUT_BARRIER - barrier after which signal is fired.
  DELIVERY_LENGTH - length of delivery output signal.
*/

module spiking_neuron_2in(clk, rst, in1, in2, out);
  
  parameter NEURON_LEVEL = -1;
  parameter NEURON_ID = -1; // id inside level
  parameter SILENT = 1;
  
  parameter INPUTS_COUNT = 2;
  
  parameter INT_WIDTH;
  parameter INT_MSB = INT_WIDTH - 1;
  parameter INT_MAX = (1 << INT_WIDTH) - 1;
  
  parameter OVERFLOW_WIDTH = 2; // upper rounding of log2 of arguments count
  
  parameter IN1_WEIGHT = INT_MAX / INPUTS_COUNT; 
  parameter IN2_WEIGHT = INT_MAX / INPUTS_COUNT;
  parameter COMPENSATION_WEIGHT = INT_MAX * INPUTS_COUNT; 
  parameter OUT_BARRIER = INT_MAX * 0.5; // fire if weighted sum is greater than it
  parameter DELIVERY_LENGTH = 1;
  
  input wire clk;
  input wire rst;
  
  input wire in1;
  input wire in2;
  
  output reg out;
  
  parameter SIGN_WIDTH = 1;
  parameter SUM_WIDTH = INT_WIDTH * 2 + OVERFLOW_WIDTH + SIGN_WIDTH;
  parameter SUM_MSB = SUM_WIDTH - 1;
  
  parameter SIGN_MSB = SUM_MSB;
  parameter SIGN_LSB = SUM_MSB - SIGN_WIDTH + 1;
  parameter OVERFLOW_MSB = SIGN_LSB - 1;
  parameter OVERFLOW_LSB = OVERFLOW_MSB - OVERFLOW_WIDTH + 1;
  parameter SUM_NEEDED_MSB = OVERFLOW_LSB - 1;
  parameter SUM_NEEDED_LSB = SUM_NEEDED_MSB - INT_WIDTH + 1;
  
  // note:
  // [                             sum                        ]
  // [sign      ] [overflow      ] [sum_needed] [not used     ]
  //  SIGN_WIDTH   OVERFLOW_WIDTH   INT_WIDTH    INT_WIDTH * 2 
  reg [SUM_MSB:0] sum;
  
  //initial $display("SIGN[%d:%d] OVERFLOW[%d:%d] NEEDED[%d:%d]", 
  //    SIGN_MSB, SIGN_LSB, OVERFLOW_MSB, OVERFLOW_LSB, SUM_NEEDED_MSB, SUM_NEEDED_LSB);
  
  parameter STATE_SIZE = 4;
  parameter STATE_MSB = STATE_SIZE - 1;
  parameter STATE_GOOD_MAX = 4;
  parameter STATE_NULL = 15;
  
  function [INT_MSB:0] calc_time_weight;
    input [STATE_MSB:0] state; // if little it is a time from getting an impulse
    begin
      case (state)
        0 : calc_time_weight = INT_MAX * 0.7;
        1 : calc_time_weight = INT_MAX * 1;
        2 : calc_time_weight = INT_MAX * 0.6;
        3 : calc_time_weight = INT_MAX * 0.3;
        4 : calc_time_weight = INT_MAX * 0.1;  
        default : calc_time_weight = 0;   
      endcase
    end
  endfunction
  
  reg [STATE_MSB:0] state_out;
  
  reg [STATE_MSB:0] state1;
  reg [STATE_MSB:0] state2;

  reg is_negative;
  reg is_overflow;
  reg [INT_MSB:0] needed;
  
  reg [STATE_MSB:0] delivery_in;
  
  always @(posedge rst)
  begin
    if (!SILENT) $display("Start!!!");
    delivery_in <= 0;
    state_out <= STATE_NULL;
    state1 <= STATE_NULL;
    state2 <= STATE_NULL;
    out <= 0;
  end
  
  // Relatively useless debug element
  always @(posedge clk)
  begin
    if (!SILENT) $display("++++ New clk! ++++");
  
    state_out = state_out < STATE_GOOD_MAX ? state_out + 1 : STATE_NULL;
    state1 = (in1 == 1) ? 0 : (state1 < STATE_GOOD_MAX ? state1 + 1 : STATE_NULL);
    state2 = (in2 == 1) ? 0 : (state2 < STATE_GOOD_MAX ? state2 + 1 : STATE_NULL);
    
    if (delivery_in != 0)
    begin
      if (!SILENT) $display("Delay shoot...");
      if (delivery_in == 1)
        begin
          if (!SILENT) $display("Fire!!!");
          out = 1;
    	   end;
      delivery_in = delivery_in - 1;
    end
    else
    begin
      out = 0;
    
      sum = - calc_time_weight(state_out) * COMPENSATION_WEIGHT
            + calc_time_weight(state1) * IN1_WEIGHT 
            + calc_time_weight(state2) * IN2_WEIGHT;
    
      if (!SILENT) $display(
          "State: raw_sum = -%3d*%3d + %3d*%3d + %3d*%3d", 
          calc_time_weight(state_out), COMPENSATION_WEIGHT, 
          calc_time_weight(state1), IN1_WEIGHT, 
          calc_time_weight(state2), IN2_WEIGHT);
      
      is_negative = (sum[SIGN_MSB:SIGN_LSB] != 0);
      is_overflow = (sum[OVERFLOW_MSB:OVERFLOW_LSB] != 0);
      needed = sum[SUM_NEEDED_MSB:SUM_NEEDED_LSB];
  
      sum = (is_negative ? 0 : (is_overflow ? INT_MAX : needed));
    
      if (!SILENT) $display("State: st1=%d, st2=%d, st_out=%d, sum=%d", state1, state2, state_out, sum);
      
      if (sum > OUT_BARRIER)
      begin
        if (!SILENT) $display("Shoot soon");
        state_out = 0;  
        if (DELIVERY_LENGTH == 0)
          out = 1;
        else
          delivery_in = DELIVERY_LENGTH;
      end;
    end;    
  end
  
endmodule