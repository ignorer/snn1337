/*
Consequent version of sort5, 
i.e. data is transmitted over one port,
one field per clk signal.
*/

module sort(clk, rst, cmd, in_data, out_data);
  parameter MAX_LEN = 5;
  parameter INT_WIDTH = 8;
  parameter INT_MSB = INT_WIDTH - 1;
  
  initial $display("Module sort initiated with INT_WIDTH = %d, MAX_LEN = %d", INT_WIDTH, MAX_LEN);

  input wire clk;
  input wire rst;
  
  input wire [0:1] cmd;

  input wire [INT_MSB:0] in_data;
  output reg [INT_MSB:0] out_data;
  
  
  int in_buff_counter;
  logic [0:4][INT_MSB:0] in_buff;
  logic [0:4][INT_MSB:0] sorted_buff;
  
  sort5 #(.INT_WIDTH(INT_WIDTH)) SORT5(
    .clk(clk),
    .rst(rst),
    .in_data(in_buff),
    .out_data(sorted_buff)
  );

  int total_counter;
  logic [0:MAX_LEN - 1][INT_MSB:0] total;
  
  assign out_data = total[total_counter];
  
  always_ff @( posedge clk )
  begin
    if (rst)
    begin
      in_buff_counter <= '0;
      total_counter <= '0;
    end
    else
    begin
      case (cmd)
        2'b01 :
        begin
          in_buff[in_buff_counter] = in_data;
          in_buff_counter = in_buff_counter + 1;
        end
        2'b10 :
        begin
          total_counter = -1;
          total[0:4] = sorted_buff;
          // copy buff to total
        end
        2'b11 :
        begin
          total_counter = total_counter + 1;
        end
      endcase
    end
  end

endmodule

