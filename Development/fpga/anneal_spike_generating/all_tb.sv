`define assert_equal(expected, got) \
  if (expected !== got) begin \
    $display("ASSERTION FAILED in %m: got %d != %d expected", got, expected); \
    $stop; \
  end

module all_tb;
  
  parameter five = 5;
  
  logic clk;
  logic rst;
  
  reg [19:0] rand_out;
  
  random_lfsr random(.clk(clk), .rst(rst), .out(rand_out));
 
  
  integer sum[0:17];
 
  reg signed[4:0] a;
  reg signed[7:0] b;
  reg signed[20:0] c;
  reg [9:0] d;
 
  initial
  begin
    c = 0;
    a = 11;
    b = -15;
    c += a * b;
    $display(c);
    $display(a * b);
    d = 1023;
    $display("%b", d);
    $display((d == -1));
    d <<= 5;
    $display("%b", d);
    $display((d == -1));
    a = {2'b11, 2'b00};
    $display(a);
    
    // $display(c[0], c[1], c[2]);
  end;
  
endmodule



