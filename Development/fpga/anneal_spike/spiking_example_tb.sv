`define assert_equal(expected, got) \
  if (expected !== got) begin \
    $display("ASSERTION FAILED in %m: got %d != %d expected", got, expected); \
    $stop; \
  end

module example_tb;
  parameter INT_WIDTH = 4;
  parameter INT_MSB = INT_WIDTH - 1;
  
  parameter FLOAT_WIDTH = INT_WIDTH * 2;
  parameter FLOAT_MSB = FLOAT_WIDTH - 1;
  
  logic clk;
  logic rst;

  logic [INT_MSB:0]  addr;
  logic [INT_MSB:0]  cmd;
  logic signed[FLOAT_MSB:0]  weight;

  logic [1:0] a, b;
  logic [1:0] c;
  
  reg neuron1_in1, neuron1_in2, neuron1_out;
  
  spiking_neuron_2in #(
    .NEURON_ID(1),
    .SILENT(0),
    .INT_WIDTH(INT_WIDTH),
    .CMD_WIDTH(INT_WIDTH),
    .ADDR_WIDTH(INT_WIDTH)
  ) neuron1(
    .clk(clk),
    .rst(rst),
    .addr(addr),
    .cmd(cmd),
    .cmd_arg(weight),
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
 
  task set_weight;
    input [INT_MSB:0] addr_;
    input [INT_MSB:0] cmd_;
    input [FLOAT_MSB:0] weight_; 
    begin
      addr = addr_;
      cmd = cmd_;
      weight = weight_;
      #20ns;
      addr = -1;
      cmd = 0;
    end
  endtask
 
  initial begin
    rst = 0;
    #10ns;
    clk = 0;
    rst = 1;
    neuron1_in1 = 0;
    neuron1_in2 = 0;
    #20ns;
    rst = 0;
    #10000ns; // seems to be enough for everything to finish
    $display("SUCCESS");
    $stop;
  end

  initial 
  begin
    #40ns;
    #5ns;
    set_weight(1, 1, 7);
    set_weight(1, 2, 7); 
    
  end

  initial 
  begin  
    #80ns;
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
    #250ns;
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
