/*
Module spiking_neural_network_xor

It is a simple module that counts XOR function
Uses neuron_example_simple inside
*/

module spiking_neural_network_xor(clk, rst, addr, cmd, cmd_arg, in, out);
  parameter INT_WIDTH = 4;
  parameter INT_MSB = INT_WIDTH - 1;
  parameter INT_MAX = (1 << INT_WIDTH) - 1;
  parameter WEIGHT_ONE = (1 << INT_WIDTH);
  
  parameter FLOAT_WIDTH = INT_WIDTH * 2;
  
  parameter ADDR_WIDTH = 3;
  parameter CMD_WIDTH = 3;
  localparam CMD_SET_DELIVERY_TIME = 2 + 1;
  localparam CMD_SET_BIAS = 2 + 2;
  localparam CMD_CLEAR = (1 << CMD_WIDTH) - 3;
  localparam CMD_SET_INPUT_TRAIN_LENGTH = (1 << CMD_WIDTH) - 4;
  localparam CMD_SET_INPUT_TRAIN_FREQUENCY = (1 << CMD_WIDTH) - 5;
  
  parameter SILENT = 1;
  
  parameter MAX_TIME = 35;
  
  input wire clk;
  input wire rst;
  wire neurons_rst;
  
  input wire [2:1] in;
  
  output reg [1:1] out;
  
  input reg [ADDR_WIDTH - 1 : 0] addr;
  input reg [CMD_WIDTH - 1 : 0] cmd;
  input reg [FLOAT_WIDTH - 1 : 0] cmd_arg;
  
  reg[31:0] counter;
  
  reg[FLOAT_WIDTH - 1 : 0] input_train_frequency[2:1];
  reg[FLOAT_WIDTH - 1 : 0] input_train_actual_frequency[2:1];
  reg[INT_WIDTH - 1 : 0] input_train_length;
  reg[INT_WIDTH - 1 : 0] input_train_actual_length;
    
  reg [2:1] neuron_0_out;
    
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
      .in1(neuron_0_out[1]), .in2(neuron_0_out[2]), .out(neuron_1_out));
  
  spiking_neuron_2in #(
    .NEURON_ID(2), 
    .INT_WIDTH(INT_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .CMD_WIDTH(CMD_WIDTH), .SILENT(0 | SILENT)
  ) neuron_2(.clk(clk), .rst(rst), .addr(addr), .cmd(cmd), .cmd_arg(cmd_arg), 
      .in1(neuron_0_out[1]), .in2(neuron_0_out[2]), .out(neuron_2_out));
  
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
    neuron_0_out = 0;
    
    for (int i = 1; i <= 2; i++)
    begin
      input_train_frequency[i] = 0;
      input_train_actual_frequency[i] = (1 << INT_WIDTH);
    end
    input_train_length = 1;
    input_train_actual_length = input_train_length;
  end
  
  always @(posedge clk)
  begin
    if (!SILENT) $display("Neural network: New clk! (%3d)", counter);
    
    if (cmd == CMD_CLEAR)
    begin
      counter = 0;
      out = 1'b z;
      for (int i = 1; i <= 2; i++)
      begin
        input_train_actual_frequency[i] = (1 << INT_WIDTH);
      end
      input_train_actual_length = input_train_length;
    end
    
    if (cmd == CMD_SET_INPUT_TRAIN_LENGTH)
      input_train_length = cmd_arg[INT_WIDTH - 1 : 0];
      
    if (cmd == CMD_SET_INPUT_TRAIN_FREQUENCY)
      input_train_frequency[addr] = cmd_arg;
      
    if (cmd == 0)
    begin
      if (input_train_actual_length > 0)
      begin
        for (int i = 1; i <= 2; i++)
        begin
          if (input_train_actual_frequency[i] >= (1 << INT_WIDTH))
          begin
            if (!SILENT) $display("Neural network: Start argument pulses");
            neuron_0_out[i] = in[i];
            input_train_actual_frequency[i] = 
                input_train_actual_frequency[i][INT_WIDTH - 1 : 0];
          end
          else
          begin
            if (!SILENT) $display("Neural network: Finish argument pulses");
            neuron_0_out[i] = 0;
          end 
          input_train_actual_frequency[i] += input_train_frequency[i];
        end 
        input_train_actual_length -= 1;
      end
      else
      begin
        neuron_0_out = 0;
      end
        
         
      if (counter == MAX_TIME)
      begin    
        if (out === 1'b z)
        begin
          out = 0;
          counter = counter - 1;
        end
      end
      if (out === 1'b z && (neuron_6_out || neuron_7_out))
      begin
        out = neuron_6_out ? 1 : 0;
      end
    counter = counter + 1;
    end    
  end
  
  
endmodule
