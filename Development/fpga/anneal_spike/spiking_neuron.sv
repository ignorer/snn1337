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
  NEURON_ID - neuron id (neuron addr)
  OVERFLOW_WIDTH - Upper rounding of log_2 of arguments count, 
    defines count of additional bits for integer overflow.
  OUT_BARRIER - barrier after which signal is fired.
  DELIVERY_LENGTH - length of delivery output signal.
*/

module spiking_neuron_2in(clk, rst, addr, cmd, cmd_arg, in1, in2, out);
  
  parameter NEURON_ID = -1;
  parameter SILENT = 1;
  
  parameter INPUTS_COUNT = 2;
  
  parameter ADDR_WIDTH = 8;
  localparam ADDR_MSB = ADDR_WIDTH - 1;
  
  parameter CMD_WIDTH = 8;
  localparam CMD_MSB = CMD_WIDTH - 1;
  localparam CMD_SET_DELIVERY_TIME = (1 << CMD_WIDTH) - 1;
  localparam CMD_SET_BIAS = (1 << CMD_WIDTH) - 2;
  localparam CMD_CLEAR = (1 << CMD_WIDTH) - 3;
  
  // int and float[0; 1) width
  parameter INT_WIDTH = 8;
  localparam INT_MSB = INT_WIDTH - 1;
  localparam INT_MAX = (1 << INT_WIDTH) - 1;
  
  // float with fixed point
  localparam FLOAT_WIDTH = INT_WIDTH * 2;
  localparam FLOAT_MSB = FLOAT_WIDTH - 1;
  
  parameter OVERFLOW_WIDTH = 2; // upper rounding of log2 of arguments count
  
  parameter COMPENSATION_WEIGHT = INT_MAX * INPUTS_COUNT; 
  parameter OUT_BARRIER = INT_MAX * 0.5; // fire if weighted sum is greater than it
  
  
  input wire clk;
  input wire rst;
  
  input wire [ADDR_MSB:0] addr;
  input wire [CMD_MSB:0] cmd;
  input wire [FLOAT_MSB:0] cmd_arg;
  
  reg signed[FLOAT_MSB:0] in1_weight;
  reg signed[FLOAT_MSB:0] in2_weight;
  reg signed[FLOAT_MSB:0] bias_weight;
  
  reg [INT_MSB:0] delivery_time;
  
  input wire in1;
  input wire in2;
  
  output reg out;
  
  localparam SIGN_WIDTH = 1;
  localparam SUM_WIDTH = INT_WIDTH * 2 + OVERFLOW_WIDTH + SIGN_WIDTH;
  localparam SUM_MSB = SUM_WIDTH - 1;
  
  localparam SIGN_MSB = SUM_MSB;
  localparam SIGN_LSB = SUM_MSB - SIGN_WIDTH + 1;
  localparam OVERFLOW_MSB = SIGN_LSB - 1;
  localparam OVERFLOW_LSB = OVERFLOW_MSB - OVERFLOW_WIDTH + 1;
  localparam SUM_NEEDED_MSB = OVERFLOW_LSB - 1;
  localparam SUM_NEEDED_LSB = SUM_NEEDED_MSB - INT_WIDTH + 1;
  
  // note:
  // [                             sum                        ]
  // [sign      ] [overflow      ] [sum_needed] [not used     ]
  //  SIGN_WIDTH   OVERFLOW_WIDTH   INT_WIDTH    INT_WIDTH * 2 
  reg signed[SUM_MSB:0] sum;
  
  localparam SUM_PART_WIDTH = FLOAT_WIDTH + INT_WIDTH;
  localparam SUM_PART_MSB = SUM_PART_WIDTH - 1;
  
  // note:
  // [        sum part raw         ]
  //          3 * INT_WIDTH
  // [ effective part ] [ not used ]
  //     FLOAT_WIDTH      INT_WIDTH
  // reg signed[SUM_PART_MSB:0] sum_part_1;
  // reg signed[SUM_PART_MSB:0] sum_part_2;
    
  //initial $display("SIGN[%d:%d] OVERFLOW[%d:%d] NEEDED[%d:%d]", 
  //    SIGN_MSB, SIGN_LSB, OVERFLOW_MSB, OVERFLOW_LSB, SUM_NEEDED_MSB, SUM_NEEDED_LSB);
  
  parameter STATE_SIZE = 4;
  localparam STATE_MSB = STATE_SIZE - 1;
  parameter STATE_GOOD_MAX = 4;
  parameter STATE_NULL = 15;
  
  function signed[INT_MSB + 1:0] calc_time_weight;
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
    in1_weight <= INT_MAX / INPUTS_COUNT;
    in2_weight <= INT_MAX / INPUTS_COUNT;
    bias_weight <= 0;
    delivery_time <= 1;
  end
  

  always @(posedge clk)
  begin
    if (!SILENT) $display("Neuron #%2d: get cmd=%d, cmd_arg=%d", 
        NEURON_ID,  cmd, cmd_arg);
    if (cmd != 0 && (addr == NEURON_ID || cmd == CMD_CLEAR))
    begin 
      case (cmd)
        1 : in1_weight = cmd_arg;
        2 : in2_weight = cmd_arg;  
        CMD_SET_BIAS : bias_weight = cmd_arg;
        CMD_SET_DELIVERY_TIME : delivery_time = cmd_arg[INT_MSB:0];
        CMD_CLEAR : 
        begin
          delivery_in <= 0;
          state_out <= STATE_NULL;
          state1 <= STATE_NULL;
          state2 <= STATE_NULL;
          out <= 0;
        end
        default : ;   
      endcase
    end
      
    if (cmd == 0)
    begin
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
        
        sum = -calc_time_weight(state_out) * COMPENSATION_WEIGHT;
        sum += bias_weight * (1 << INT_WIDTH);
        sum += calc_time_weight(state1) * in1_weight;
        sum += calc_time_weight(state2) * in2_weight;
      
        if (!SILENT) $display(
            "State: raw_sum %d = -%3d*%3d + %3d*%3d (b) + %3d*%3d (in1) + %3d*%3d (in2)", 
            sum,
            calc_time_weight(state_out), COMPENSATION_WEIGHT, 
            (1 << INT_WIDTH), bias_weight,
            calc_time_weight(state1), in1_weight, 
            calc_time_weight(state2), in2_weight);
        
        is_negative = (sum[SIGN_MSB:SIGN_LSB] != 0);
        is_overflow = (sum[OVERFLOW_MSB:OVERFLOW_LSB] != 0);
        needed = sum[SUM_NEEDED_MSB:SUM_NEEDED_LSB];
    
        sum = (is_negative ? 0 : (is_overflow ? INT_MAX : needed));
      
        if (!SILENT) $display("State: st1=%d, st2=%d, st_out=%d, sum=%d", state1, state2, state_out, sum);
        
        if (sum > OUT_BARRIER)
        begin
          if (!SILENT) $display("Shoot soon");
          state_out = 0;  
          if (delivery_time == 0)
            out = 1;
          else
            delivery_in = delivery_time;
        end
      end
    end    
  end
  
endmodule