/*
Very fast sorter for 5 integers.
Uses data-flow model.
*/

module sort5(clk, rst, in_data, out_data);
  parameter INT_WIDTH = 8;
  parameter INT_MSB = INT_WIDTH - 1;
  
  initial $display("Module sort5 initiated with INT_WIDTH = %d, INT_MSB = %d", INT_WIDTH, INT_MSB);

  input wire clk;
  input wire rst;

  input wire [0:4][INT_MSB:0] in_data;
  output reg [0:4][INT_MSB:0] out_data;

  
  logic [0:4][INT_MSB:0] order;

  function [INT_MSB:0] extract;
    input [0:4][INT_MSB:0] data;
    input [0:4][INT_MSB:0] order;
    input [0:3] i;
    
    begin
      extract = 
          data[0] * (order[0] == i) + 
          data[1] * (order[1] == i) + 
          data[2] * (order[2] == i) + 
          data[3] * (order[3] == i) + 
          data[4] * (order[4] == i);   
    end
  endfunction  
  
  
  assign order[0] = (in_data[1] < in_data[0]) + (in_data[2] < in_data[0]) + (in_data[3] < in_data[0]) + (in_data[4] < in_data[0]);
  assign order[1] = (in_data[0] <= in_data[1]) + (in_data[2] < in_data[1]) + (in_data[3] < in_data[1]) + (in_data[4] < in_data[1]);
  assign order[2] = (in_data[0] <= in_data[2]) + (in_data[1] <= in_data[2]) + (in_data[3] < in_data[2]) + (in_data[4] < in_data[2]);
  assign order[3] = (in_data[0] <= in_data[3]) + (in_data[1] <= in_data[3]) + (in_data[2] <= in_data[3]) + (in_data[4] < in_data[3]);
  assign order[4] = (in_data[0] <= in_data[4]) + (in_data[1] <= in_data[4]) + (in_data[2] <= in_data[4]) + (in_data[3] <= in_data[4]);
  
  assign out_data[0] = extract(in_data, order, 0);
  assign out_data[1] = extract(in_data, order, 1);
  assign out_data[2] = extract(in_data, order, 2);
  assign out_data[3] = extract(in_data, order, 3);
  assign out_data[4] = extract(in_data, order, 4);

endmodule
