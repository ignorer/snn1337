/*
Module spiking_neural_network_xor

It is a simple module that counts XOR function
Uses neuron_example_simple inside
*/

module spiking_neural_network_xor(clk, rst, addr, cmd, cmd_arg, in1, in2, out, out_time);
  parameter INT_WIDTH = 4;
  parameter INT_MSB = INT_WIDTH - 1;
  parameter INT_MAX = (1 << INT_WIDTH) - 1;
  parameter WEIGHT_ONE = (1 << INT_WIDTH);
  
  parameter FLOAT_WIDTH = INT_WIDTH * 2;
  
  parameter ADDR_WIDTH = 3;
  parameter CMD_WIDTH = 3;
  localparam CMD_SET_DELIVERY_TIME = (1 << CMD_WIDTH) - 1;
  localparam CMD_SET_BIAS = (1 << CMD_WIDTH) - 2;
  localparam CMD_CLEAR = (1 << CMD_WIDTH) - 3;
  
  parameter SILENT = 1;
  
  parameter MAX_TIME = 35;
  
  input wire clk;
  input wire rst;
  wire neurons_rst;
  
  input wire in1;
  input wire in2;
  
  output reg out;
  output reg [31:0] out_time;
  
  input reg [ADDR_WIDTH - 1 : 0] addr;
  input reg [CMD_WIDTH - 1 : 0] cmd;
  input reg [FLOAT_WIDTH - 1 : 0] cmd_arg;
  
  reg[31:0] counter;
  
  reg neuron_0_1_out;
  reg neuron_0_2_out;
    
  wire neuron_1_out;
  wire neuron_2_out;
  wire neuron_3_out;
  wire neuron_4_out;
  wire neuron_5_out;
  wire neuron_6_out;
  wire neuron_7_out;
    
  spiking_neuron_2in #(
    .NEURON_ID(1), 
    .INT_WIDTH(INT_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .CMD_WIDTH(CMD_WIDTH), .SILENT(0 | SILENT)
  ) neuron_1(.clk(clk), .rst(rst), .addr(addr), .cmd(cmd), .cmd_arg(cmd_arg), 
      .in1(neuron_0_1_out), .in2(neuron_0_2_out), .out(neuron_1_out));
  
  spiking_neuron_2in #(
    .NEURON_ID(2), 
    .INT_WIDTH(INT_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .CMD_WIDTH(CMD_WIDTH), .SILENT(0 | SILENT)
  ) neuron_2(.clk(clk), .rst(rst), .addr(addr), .cmd(cmd), .cmd_arg(cmd_arg), 
      .in1(neuron_0_1_out), .in2(neuron_0_2_out), .out(neuron_2_out));
  
  spiking_neuron_2in #(
    .NEURON_ID(4), 
    .INT_WIDTH(INT_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .CMD_WIDTH(CMD_WIDTH), .SILENT(0 | SILENT)
  ) neuron_4(.clk(clk), .rst(rst), .addr(addr), .cmd(cmd), .cmd_arg(cmd_arg),
      .in1(neuron_1_out), .in2(neuron_2_out), .out(neuron_4_out));
    
  spiking_neuron_2in #(
    .NEURON_ID(3), 
    .INT_WIDTH(INT_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .CMD_WIDTH(CMD_WIDTH), .SILENT(0 | SILENT)
  ) neuron_3(.clk(clk), .rst(rst), .addr(addr), .cmd(cmd), .cmd_arg(cmd_arg),
      .in1(neuron_3_out), .in2(neuron_3_out), .out(neuron_3_out));  
  
  spiking_neuron_2in #(
    .NEURON_ID(6), 
    .INT_WIDTH(INT_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .CMD_WIDTH(CMD_WIDTH), .SILENT(0 | SILENT)
  ) neuron_6(.clk(clk), .rst(rst), .addr(addr), .cmd(cmd), .cmd_arg(cmd_arg),
      .in1(neuron_4_out), .in2(neuron_6_out), .out(neuron_6_out));  
  
  spiking_neuron_2in #(
    .NEURON_ID(5), 
    .INT_WIDTH(INT_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .CMD_WIDTH(CMD_WIDTH), .SILENT(0 | SILENT)
  ) neuron_5(.clk(clk), .rst(rst), .addr(addr), .cmd(cmd), .cmd_arg(cmd_arg),
      .in1(neuron_3_out), .in2(neuron_6_out), .out(neuron_5_out));  
  
  spiking_neuron_2in #(
    .NEURON_ID(7), 
    .INT_WIDTH(INT_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .CMD_WIDTH(CMD_WIDTH), .SILENT(0 | SILENT)
  ) neuron_7(.clk(clk), .rst(rst), .addr(addr), .cmd(cmd), .cmd_arg(cmd_arg),
      .in1(neuron_5_out), .in2(neuron_7_out), .out(neuron_7_out));  
  
  
  always @(posedge rst)
  begin
    if (!SILENT) $display("Neural network started");
    counter = -1;
    out = 1'b z;
    neuron_0_1_out = 0;
    neuron_0_2_out = 0;
    out_time = MAX_TIME;
  end
  
  always @(posedge clk)
  begin
    if (!SILENT) $display("Neural network: New clk! (%3d)", counter);
    
    if (cmd == CMD_CLEAR)
    begin
      counter = 0;
      out = 1'b z;
      out_time = MAX_TIME;
    end
      
    if (cmd == 0)
    begin
      case (counter)
        0 : 
        begin
          if (!SILENT) $display("Neural network: Start argument pulses");
          neuron_0_1_out = in1;
          neuron_0_2_out = in2;
        end
        1 : 
        begin
          if (!SILENT) $display("Neural network: Finish argument pulses");
          neuron_0_1_out = 0;
          neuron_0_2_out = 0;
        end  
        MAX_TIME :
        begin
          if (out === 1'b z)
          begin
            out = 0;
            out_time = counter + 10;
            counter = counter - 1;
          end
        end
      endcase;
      if (out === 1'b z && (neuron_6_out || neuron_7_out))
      begin
        out = neuron_6_out ? 1 : 0;
        out_time = counter;
      end
    counter = counter + 1;
    end    
  end
  
  
endmodule
