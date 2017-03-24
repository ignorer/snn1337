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
  
  reg [INT_MSB:0] neuron1_in1, neuron1_in2, neuron1_out;
  
  neuron_example_simple #(
    .NEURON_LEVEL(1),
    .NEURON_ID(1),
    .INT_WIDTH(INT_WIDTH), 
    .IN1_WEIGHT(32), 
    .IN2_WEIGHT(-16)
  ) neuron1(
    .clk(clk),
    .rst(rst),
    .in1(neuron1_in1),
    .in2(neuron1_in2),
    .out(neuron1_out)
  );
  
  reg neural_nework_in1, neural_nework_in2, neural_nework_out;
  
  neural_network_simple #(
    .INT_WIDTH(INT_WIDTH)
  ) neural_network(
    .clk(clk),
    .rst(rst),
    .in1(neural_nework_in1),
    .in2(neural_nework_in2),
    .out(neural_nework_out)
  );
  
 
  initial begin
    #1000000000ns // seems to be enough for everything to finish
    $display("SUCCESS");
    $stop;
  end
  
  
  task test_neuron1;
    input [INT_MSB:0] in1, in2; 
    begin
      automatic int expected = in1 * 2 - in2;
      if (expected < 0)
        expected = 0;
      if (expected >= 16)
        expected = 15;
      
      neuron1_in1 = in1;
      neuron1_in2 = in2;
      
      #10ns;
      
      // making debug output from neuron
      //clk = 0;
      //#10ns;
      //clk = 1;
      //#10ns;
      
      //$display("In: %d, %d; Out: %d", 
      //  neuron1_in1, neuron1_in2, neuron1_out);
      `assert_equal(expected, neuron1_out);
    end
  endtask
  
  // test neuron simple
  initial 
  begin  
    test_neuron1(1, 1); 
    test_neuron1(1, 2); 
    test_neuron1(2, 1); 
    test_neuron1(7, 5);
    test_neuron1(15, 5);    
  end
  
  task test_neural_network_simple;
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
  


  

endmodule
