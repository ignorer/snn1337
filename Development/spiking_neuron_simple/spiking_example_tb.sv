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
    .SILENT(1),
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
  
endmodule
