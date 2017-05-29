`define assert_equal(expected, got) \
  if (expected !== got) begin \
    $display("ASSERTION FAILED in %m: got %d != %d expected", got, expected); \
    $stop; \
  end

module example_network_tb;
  parameter INT_WIDTH = 4;
  parameter INT_MSB = INT_WIDTH - 1;
  parameter INT_MAX = (1 << INT_WIDTH) - 1;
  
  parameter FLOAT_WIDTH = INT_WIDTH * 2;
  
  parameter CMD_WIDTH = 3;
  parameter ADDR_WIDTH = 3;

  logic clk;
  logic rst;
  
  logic [ADDR_WIDTH - 1:0]  addr;
  logic [CMD_WIDTH - 1:0]  cmd;
  logic signed[FLOAT_WIDTH - 1:0]  weight;
  
  reg neural_nework_in1, neural_nework_in2, neural_nework_out;
  
  reg [31:0] neural_network_out_time;
  
  spiking_neural_network_xor #(
    .INT_WIDTH(INT_WIDTH),
    .SILENT(0)
  ) neural_network(
    .clk(clk),
    .rst(rst),
    .addr(addr), .cmd(cmd), .cmd_arg(weight),
    .in1(neural_nework_in1),
    .in2(neural_nework_in2),
    .out(neural_nework_out),
    .out_time(neural_network_out_time)
  );
  
  task set_weight;
    input [INT_MSB:0] addr_;
    input [INT_MSB:0] cmd_;
    input [FLOAT_WIDTH - 1:0] weight_; 
    begin
      addr = addr_;
      cmd = cmd_;
      weight = weight_;
      #20ns;
      addr = -1;
    end
  endtask
 
  always begin
    #10ns;
    //$display("---posedge---");
    clk = ~clk;
    #10ns;
    //$display("---negedge---");
    clk = ~clk;
  end
 
  initial begin
    clk = 0;
    #4000ns; // seems to be enough for everything to finish
    $display("SUCCESS");
    $stop;
  end
  
  task test_neural_network_simple;
    input in1, in2; 
    begin
      automatic int expected = in1 ^ in2;

      neural_nework_in1 = in1;
      neural_nework_in2 = in2;
      
      rst = 1;
      #20ns
      rst = 0;
      set_weight(1, 1, INT_MAX / 2.1);
      set_weight(1, 2, INT_MAX / 2.1);
      set_weight(2, 1, INT_MAX * 1.2);
      set_weight(2, 2, INT_MAX * 1.2);
      set_weight(3, 1, -INT_MAX);
      set_weight(3, 2, INT_MAX);
      
      cmd = 0;
      #20ns;
      cmd = -1;
      
      #500ns; // wait for calculations
            
      `assert_equal(expected, neural_nework_out);
    end
  endtask
  
  // test neural network
  initial
  begin
    test_neural_network_simple(0, 0);
    test_neural_network_simple(0, 1);
    test_neural_network_simple(1, 0);
    test_neural_network_simple(1, 1);    
  end

endmodule
