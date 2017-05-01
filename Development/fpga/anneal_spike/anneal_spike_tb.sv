`define assert_equal(expected, got) \
  if (expected !== got) begin \
    $display("ASSERTION FAILED in %m: got %d != %d expected", got, expected); \
    $stop; \
  end

module anneal_spike_tb;
  parameter INT_WIDTH = 4;
  parameter INT_MSB = INT_WIDTH - 1;
  parameter INT_MAX = (1 << INT_WIDTH) - 1;
  
  parameter FLOAT_WIDTH = INT_WIDTH * 2;
  
  parameter CMD_WIDTH = 3;
  parameter ADDR_WIDTH = 3;
  
  localparam CMD_SET_DELIVERY_TIME = (1 << CMD_WIDTH) - 1;
  localparam CMD_SET_BIAS = (1 << CMD_WIDTH) - 2;
  localparam CMD_CLEAR = (1 << CMD_WIDTH) - 3;

  logic clk;
  logic rst;
  
  logic [ADDR_WIDTH - 1:0]  addr;
  logic [CMD_WIDTH - 1:0]  cmd;
  logic signed[FLOAT_WIDTH - 1:0]  weight;
  
  reg neural_network_in1, neural_network_in2, neural_network_out;
  
  int neural_network_fitness;
  
  reg [31:0] neural_network_out_time;
  
  spiking_neural_network_xor #(
    .INT_WIDTH(INT_WIDTH),
    .SILENT(1)
  ) neural_network(
    .clk(clk),
    .rst(rst),
    .addr(addr), .cmd(cmd), .cmd_arg(weight),
    .in1(neural_network_in1),
    .in2(neural_network_in2),
    .out(neural_network_out),
    .out_time(neural_network_out_time)
  );
  
  reg [19:0] rand_out;
  
  random_lfsr random(.clk(clk), .rst(rst), .out(rand_out));
  
  task run_cmd;
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
    #4000000000ns; // seems to be enough for everything to finish
    $display("SUCCESS");
    $stop;
  end
  
  task test_neural_network_simple;
    input in1, in2; 
    begin
      automatic int expected = in1 ^ in2;

      neural_network_in1 = in1;
      neural_network_in2 = in2;
      
      rst = 1;
      #20ns
      rst = 0;
      run_cmd(1, 1, INT_MAX / 2.1);
      run_cmd(1, 2, INT_MAX / 2.1);
      run_cmd(2, 1, INT_MAX * 1.2);
      run_cmd(2, 2, INT_MAX * 1.2);
      run_cmd(4, 1, -INT_MAX);
      run_cmd(4, 2, INT_MAX);
      
      run_cmd(3, CMD_SET_BIAS, INT_MAX);
      run_cmd(3, CMD_SET_DELIVERY_TIME, 12);
      
      run_cmd(5, 1, INT_MAX * 0.6);
      run_cmd(5, 2, -INT_MAX * 2);
      
      run_cmd(6, 1, INT_MAX);
      run_cmd(6, 2, INT_MAX);
      run_cmd(6, CMD_SET_DELIVERY_TIME, 2);
    
      run_cmd(7, 1, INT_MAX);
      run_cmd(7, 2, INT_MAX);
      run_cmd(7, CMD_SET_DELIVERY_TIME, 2);
      
      cmd = CMD_CLEAR;
      #20ns;
      cmd = 0;
      
      while (neural_network_out === 1'b z)
        #20ns;
            
      `assert_equal(expected, neural_network_out);
    end
  endtask
  
  task xor_net_ask;
    input in1, in2; 
    begin
      neural_network_in1 = in1;
      neural_network_in2 = in2;
      
      cmd = CMD_CLEAR;
      #20ns;
      cmd = 0;
      
      // wait for calculations
      while (neural_network_out === 1'b z)
        #20ns;          
    end
  endtask
  
  task xor_net_fitness;
    begin
      neural_network_fitness = 0;
      xor_net_ask(0, 0);
      neural_network_fitness += (neural_network_out == 0);
      //$display("00 -> %d", neural_network_out);
      xor_net_ask(1, 1);
      neural_network_fitness += (neural_network_out == 0);
      //$display("11 -> %d", neural_network_out);
      xor_net_ask(0, 1);
      neural_network_fitness += (neural_network_out == 1);
      //$display("01 -> %d", neural_network_out);
      xor_net_ask(1, 0);
      neural_network_fitness += (neural_network_out == 1);
      //$display("10 -> %d", neural_network_out);
    end  
  endtask
  
  task anneal;
   parameter FLOAT_BASE = (1 << 10);
   
    int neuron_id;
    int neuron_in;
    int neuron_new_weight;
   int weights[1:7][1:2];
   int old_fitness, new_fitness, delta;
   int exp_power, exp, exp_last_member;
   int temperature;
   reg accept_change;
    begin
      rst = 1;
      #20ns
      rst = 0;
      
      weights[1][1] = INT_MAX / 2.1;
      weights[1][2] = INT_MAX / 2.1;
      weights[2][1] = INT_MAX * 1.2;
     	weights[2][2] = INT_MAX * 1.2;
      weights[4][1] = -INT_MAX;
      weights[4][2] = INT_MAX;
      
      run_cmd(3, CMD_SET_BIAS, INT_MAX);
      run_cmd(3, CMD_SET_DELIVERY_TIME, 12);
      
      weights[5][1] = INT_MAX;
      weights[5][2] = -INT_MAX;
      
      weights[6][1] = INT_MAX;
      weights[6][2] = INT_MAX;
      run_cmd(6, CMD_SET_DELIVERY_TIME, 2);
    
      weights[7][1] = INT_MAX;
      weights[7][2] = INT_MAX;
      run_cmd(7, CMD_SET_DELIVERY_TIME, 2);
      
      if (0)
      for (int id = 1; id <= 7; id++)
      begin
        for (int in = 1; in <= 2; in++)
        begin
          run_cmd(id, in, weights[id][in]);
        end
      end
         
      xor_net_fitness();
      
      old_fitness = neural_network_fitness * FLOAT_BASE;
      
      temperature = 30 * FLOAT_BASE;
      
      $display(old_fitness);
      
      for (int i = 0; i < 4300; i++)
      begin
        neuron_id = (rand_out % 7) + 1;
        #20ns;
        neuron_in = (rand_out % 2) + 1; 
        #20ns;
        neuron_new_weight = (rand_out % (INT_MAX * 4)) - INT_MAX * 2; 
        #20ns;
        run_cmd(neuron_id, neuron_in, neuron_new_weight);
        xor_net_fitness();
        new_fitness = neural_network_fitness * FLOAT_BASE;
        
        accept_change = 0;
       
       if (new_fitness > old_fitness)
         accept_change =1;
       else
       begin
         delta = old_fitness - new_fitness;
         exp_power = -delta * FLOAT_BASE / temperature;
          exp = 1 * FLOAT_BASE;
         exp_last_member = 1 * FLOAT_BASE;
         
          if (exp_power < -10)
          begin
            exp = 0;
           exp_last_member = 0;
         end      
         for (int j = 1; j < 20; j++)
         begin
           exp_last_member = exp_last_member * exp_power / FLOAT_BASE / j;
           exp += exp_last_member;
         end
         if (rand_out % FLOAT_BASE < exp)
           accept_change = 1;
         $display("e^%d =?  %d", exp_power, exp);
       end 
       if (accept_change)
       begin
         weights[neuron_id][neuron_in] = neuron_new_weight;
         old_fitness = new_fitness;
       end
       else
        begin
          run_cmd(neuron_id, neuron_in, weights[neuron_id][neuron_in]);
          #20ns;
        end         
           
        
        //$display("run_cmd(%2d, %1d, %4d), new fitness = %1d",
        //    neuron_id, neuron_in, neuron_new_weight, neural_network_fitness);
        $display("%d iteration, fitness = %d, temperature = %d", i, old_fitness, temperature);
            
        if (i % 10 == 0)
        begin
          temperature = (temperature - 1) * 0.985;
          if (temperature < 3)
            temperature = 3;
        end
     end
      
      $stop;
    end
  endtask
  
  // test neural network
  initial
  begin
    anneal();   
  end
  
endmodule


