`define assert_equal(expected, got) \
  if (expected !== got) begin \
    $display("ASSERTION FAILED in %m: got %d != %d expected", got, expected); \
    $stop; \
  end

module example_tb;
  parameter INT_WIDTH = 4;
  parameter INT_MSB = INT_WIDTH - 1;

  logic clk;
  logic rst;

  logic [1:0] a, b;
  logic [1:0] c;
  
  reg neuron1_in1, neuron1_in2, neuron1_out;
  
  spiking_neuron_2in #(
    .NEURON_LEVEL(1),
    .NEURON_ID(1),
    .INT_WIDTH(INT_WIDTH), 
    .IN1_WEIGHT(7), 
    .IN2_WEIGHT(7)
  ) neuron1(
    .clk(clk),
    .rst(rst),
    .in1(neuron1_in1),
    .in2(neuron1_in2),
    .out(neuron1_out)
  );
  /*
  reg neural_nework_in1, neural_nework_in2, neural_nework_out;
  
  neural_network_simple #(
    .INT_WIDTH(INT_WIDTH)
  ) neural_network(
    .clk(clk),
    .rst(rst),
    .in1(neural_nework_in1),
    .in2(neural_nework_in2),
    .out(neural_nework_out)
  );*/
  
 
  always begin
    #10ns;
    #1ns $display("---posedge---");
    clk = ~clk;
    #10ns;
    $display("---negedge---");
    clk = ~clk;
  end
 
  initial begin
    clk = 0;
    rst = 1;
    neuron1_in1 = 0;
    neuron1_in2 = 0;
    #30ns;
    rst = 0;
    #350ns; // seems to be enough for everything to finish
    $display("SUCCESS");
    $stop;
  end

  initial 
  begin  
    #40ns;
    #5ns;
    neuron1_in1 = 1;
    #20ns;
    neuron1_in1 = 0;
    neuron1_in2 = 1;
    #20ns;
    neuron1_in2 = 0;
    `assert_equal(0, neuron1_out);
    #20ns;
    `assert_equal(1, neuron1_out);
    #20ns;
    `assert_equal(0, neuron1_out);
  end
  
  initial 
  begin  
    #210ns;
    #5ns;
    neuron1_in1 = 1;
    neuron1_in2 = 1;
    #20ns;
    neuron1_in1 = 0;
    neuron1_in2 = 0;
    `assert_equal(0, neuron1_out);
    #20ns;
    `assert_equal(1, neuron1_out);
    #20ns;
    `assert_equal(0, neuron1_out);
  end
  
  /*task test_neural_network_simple;
    input in1, in2; 
    begin
      automatic int expected = in1 ^ in2;
      
      neural_nework_in1 = in1;
      neural_nework_in2 = in2;
      //$display("Test neural networks: %d ^ %d = ?", in1, in2);
      
      #10ns
      
      // making debug output from neuron
      //clk = 0;
      //#10ns;
      //clk = 1;
      //#10ns;
      
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
  
*/

  

endmodule
