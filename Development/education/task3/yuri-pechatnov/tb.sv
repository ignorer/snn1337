module tb;

  logic clk;
  logic rst;

  logic [7:0] symbol;
  
  logic [0:4][7:0] in_data;
  
  logic [0:4][7:0] out_data;
  
  sort5 #(.INT_WIDTH(8)) DUT(
    .clk(clk),
    .rst(rst),
    .in_data(in_data),
    .out_data(out_data)
  );
  
  logic [0:1] cmd_sort;
  logic [7:0] to_sort;
  logic [7:0] from_sort;  
  
  sort #(.INT_WIDTH(8)) SORTER(
    .clk(clk),
    .rst(rst),
    .cmd(cmd_sort),
    .in_data(to_sort),
    .out_data(from_sort)
  );
  
  initial begin
    clk = 0;
    rst = 1;
    
    in_data[0] = 5;
    in_data[1] = 3;
    in_data[2] = 4;
    in_data[3] = 1;
    in_data[4] = 4;
    
    #45ns;
    rst = 0;
    
    $display("Sorted by sort5 (ans = 1, 3, 4, 4, 5): ");
    for (integer i = 0; i < 5; i++)
    begin
      $write("%d, ", out_data[i]);
    end
    $display("");
    

    
    #4000ns
    $stop;
  end

  
  initial begin
    #65ns;
    cmd_sort = 2'b01;
    to_sort = 5;
    #20ns;
    cmd_sort = 2'b01;
    to_sort = 4;
    #20ns;
    cmd_sort = 2'b01;
    to_sort = 3;
    #20ns;
    cmd_sort = 2'b01;
    to_sort = 2;
    #20ns;
    cmd_sort = 2'b01;
    to_sort = 1;
    #20ns;
    cmd_sort = 2'b10;
    #40ns;
    $display("Sorted by consequent sort (ans = 1, 2, 3, 4, 5):");
    cmd_sort = 2'b11;
    #10ns;
    $write("%d, ", from_sort);
    #10ns;
    cmd_sort = 2'b11;
    #10ns;
    $write("%d, ", from_sort);
    #10ns;
    cmd_sort = 2'b11;
    #10ns;
    $write("%d, ", from_sort);
    #10ns;
    cmd_sort = 2'b11;
    #10ns;
    $write("%d, ", from_sort);
    #10ns;
    cmd_sort = 2'b11;
    #10ns;
    $write("%d, ", from_sort);
    #10ns;
  end
  
  always begin
    #10ns;
    clk = ~clk;
    //#100000ns;
  end
  

endmodule
