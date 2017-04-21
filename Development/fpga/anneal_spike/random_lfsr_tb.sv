`define assert_equal(expected, got) \
  if (expected !== got) begin \
    $display("ASSERTION FAILED in %m: got %d != %d expected", got, expected); \
    $stop; \
  end

module random_lfsr_tb;
  
  logic clk;
  logic rst;
  
  reg [19:0] rand_out;
  
  random_lfsr random(.clk(clk), .rst(rst), .out(rand_out));
 
  
  integer sum[0:17];
 
  initial 
  begin 
    rst = 0;
    #40ns;
    rst = 1;
    #40ns;
    rst = 0;
    #40ns;
    clk = 0; 
    for (int i = 0; i < 17; i++)
    begin
     sum[i] = 0;
   end
    for (int i = 0; i < 10000; i++)
    begin
      #10ns;
      // $display("---posedge---");
      clk = ~clk;
      #10ns;
      // $display("---negedge---");
      clk = ~clk;
      sum[rand_out % 17] += 1;
    end
    for (int i = 0; i < 17; i++)
    begin 
      assert(450 < sum[i] && sum[i] < 700);
      // $display(sum[i]);
    end
    $stop;
  end;
  
endmodule


