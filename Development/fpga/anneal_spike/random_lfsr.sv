

/*
Module random_lfsr
Using 49 raw bits
Idea have been got from:
https://electronics.stackexchange.com/questions/30521/random-bit-sequence-using-verilog

*/

module random_lfsr(clk, rst, out);
  
  parameter MOD_BIG = 1000000009;
  parameter RET_INT_SIZE = 20;
  
  input wire clk;
  input wire rst;
  
  output reg [RET_INT_SIZE - 1:0] out;
  
  reg [48 : 0] rand_bits;
  reg [48 : 0] moded_bits;
  
  always @(posedge rst)
  begin
    rand_bits <= 1234567890;
  end
  
  // gen rand
  always @(posedge clk)
  begin
    rand_bits = { rand_bits[47:0], !(rand_bits[48] ^ rand_bits[39]) };
    moded_bits = (rand_bits % MOD_BIG);
    out = moded_bits[RET_INT_SIZE - 1:0];
  end
  
endmodule
