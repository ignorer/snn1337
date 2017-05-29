`define assert_equal(expected, got) \
  if (expected !== got) begin \
    $display("ASSERTION FAILED in %m: got %d != %d expected", got, expected); \
    $stop; \
  end

module example_network_tb;
  parameter INT_WIDTH = 4;
  parameter INT_MSB = INT_WIDTH - 1;

  logic clk;
  logic rst;
  
  reg neural_nework_in1, neural_nework_in2, neural_nework_out;
  
  spiking_neural_network_xor #(
    .INT_WIDTH(INT_WIDTH)
  ) neural_network(
    .clk(clk),
    .rst(rst),
    .in1(neural_nework_in1),
    .in2(neural_nework_in2),
    .out(neural_nework_out)
  );
 
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
      
      #500ns // wait for calculations
            
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
