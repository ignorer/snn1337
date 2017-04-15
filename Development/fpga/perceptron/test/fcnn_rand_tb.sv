module neuron2in(clk, rst, in0, in1, out);

parameter W0 = 0;
parameter W1 = 0;

input wire clk;
input wire rst;

input signed [15:0] in0;
input signed [15:0] in1;

output reg signed [15:0] out;

reg signed [31:0] x;
reg [31:0] abs_x;
reg [31:0] y;
always @* begin
    x = in0 * W0 / 1000 + in1 * W1 / 1000;
    abs_x = x < 0 ? -x : x;
    if (abs_x >= 5000) y = 1000;
    else if (abs_x >= 2375) y = 31 * abs_x / 1000 + 844;
    else if (abs_x >= 1000) y = 125 * abs_x / 1000 + 625;
    else if (abs_x >= 0) y = 250 * abs_x / 1000 + 500;
    out = y;
end

endmodule

module neuron4in(clk, rst, in0, in1, in2, in3, out);

parameter W0 = 0;
parameter W1 = 0;
parameter W2 = 0;
parameter W3 = 0;

input wire clk;
input wire rst;

input signed [15:0] in0;
input signed [15:0] in1;
input signed [15:0] in2;
input signed [15:0] in3;

output reg signed [15:0] out;

reg signed [31:0] x;
reg [31:0] abs_x;
reg [31:0] y;
always @* begin
    x = in0 * W0 / 1000 + in1 * W1 / 1000 + in2 * W2 / 1000 + in3 * W3 / 1000;
    abs_x = x < 0 ? -x : x;
    if (abs_x >= 5000) y = 1000;
    else if (abs_x >= 2375) y = 31 * abs_x / 1000 + 844;
    else if (abs_x >= 1000) y = 125 * abs_x / 1000 + 625;
    else if (abs_x >= 0) y = 250 * abs_x / 1000 + 500;
    out = y;
end

endmodule

module layer2in4out(clk, rst, in0, in1, out0, out1, out2, out3);

parameter W0TO0 = 0;
parameter W0TO1 = 0;
parameter W0TO2 = 0;
parameter W0TO3 = 0;
parameter W1TO0 = 0;
parameter W1TO1 = 0;
parameter W1TO2 = 0;
parameter W1TO3 = 0;

input wire clk;
input wire rst;

input signed [15:0] in0;
input signed [15:0] in1;

output signed [15:0] out0;
output signed [15:0] out1;
output signed [15:0] out2;
output signed [15:0] out3;

neuron2in #(.W0(W0TO0), .W1(W1TO0)) neuron0(.clk(clk), .rst(rst), .in0(in0), .in1(in1), .out(out0));
neuron2in #(.W0(W0TO1), .W1(W1TO1)) neuron1(.clk(clk), .rst(rst), .in0(in0), .in1(in1), .out(out1));
neuron2in #(.W0(W0TO2), .W1(W1TO2)) neuron2(.clk(clk), .rst(rst), .in0(in0), .in1(in1), .out(out2));
neuron2in #(.W0(W0TO3), .W1(W1TO3)) neuron3(.clk(clk), .rst(rst), .in0(in0), .in1(in1), .out(out3));

endmodule

module layer4in2out(clk, rst, in0, in1, in2, in3, out0, out1);

parameter W0TO0 = 0;
parameter W0TO1 = 0;
parameter W1TO0 = 0;
parameter W1TO1 = 0;
parameter W2TO0 = 0;
parameter W2TO1 = 0;
parameter W3TO0 = 0;
parameter W3TO1 = 0;

input wire clk;
input wire rst;

input signed [15:0] in0;
input signed [15:0] in1;
input signed [15:0] in2;
input signed [15:0] in3;

output signed [15:0] out0;
output signed [15:0] out1;

neuron4in #(.W0(W0TO0), .W1(W1TO0), .W2(W2TO0), .W3(W3TO0)) neuron0(.clk(clk), .rst(rst), .in0(in0), .in1(in1), .in2(in2), .in3(in3), .out(out0));
neuron4in #(.W0(W0TO1), .W1(W1TO1), .W2(W2TO1), .W3(W3TO1)) neuron1(.clk(clk), .rst(rst), .in0(in0), .in1(in1), .in2(in2), .in3(in3), .out(out1));

endmodule

module layer2in2out(clk, rst, in0, in1, out0, out1);

parameter W0TO0 = 0;
parameter W0TO1 = 0;
parameter W1TO0 = 0;
parameter W1TO1 = 0;

input wire clk;
input wire rst;

input signed [15:0] in0;
input signed [15:0] in1;

output signed [15:0] out0;
output signed [15:0] out1;

neuron2in #(.W0(W0TO0), .W1(W1TO0)) neuron0(.clk(clk), .rst(rst), .in0(in0), .in1(in1), .out(out0));
neuron2in #(.W0(W0TO1), .W1(W1TO1)) neuron1(.clk(clk), .rst(rst), .in0(in0), .in1(in1), .out(out1));

endmodule

module network(clk, rst, in0, in1, out0, out1);

input wire clk;
input wire rst;

input signed [15:0] in0;
input signed [15:0] in1;

output signed [15:0] out0;
output signed [15:0] out1;

wire[15:0] con0[0:3];
wire[15:0] con1[0:1];

layer2in4out #(.W0TO0(-475), .W0TO1(-682), .W0TO2(-443), .W0TO3(-80), .W1TO0(-357), .W1TO1(37), .W1TO2(-475), .W1TO3(952)) layer0(.clk(clk), .rst(rst), .in0(in0), .in1(in1), .out0(con0[0]), .out1(con0[1]), .out2(con0[2]), .out3(con0[3]));
layer4in2out #(.W0TO0(466), .W0TO1(-768), .W1TO0(-226), .W1TO1(257), .W2TO0(-749), .W2TO1(967), .W3TO0(-113), .W3TO1(579)) layer1(.clk(clk), .rst(rst), .in0(con0[0]), .in1(con0[1]), .in2(con0[2]), .in3(con0[3]), .out0(con1[0]), .out1(con1[1]));
layer2in2out #(.W0TO0(588), .W0TO1(-276), .W1TO0(-167), .W1TO1(169)) layer2(.clk(clk), .rst(rst), .in0(con1[0]), .in1(con1[1]), .out0(out0), .out1(out1));

endmodule

`define assert_close(expected, got, eps) \
if ((expected > got && expected > got + eps) || (expected < got && expected + eps < got)) begin \
    $display("TEST FAILED in %m: got %d, expected %d", got, expected); \
    $stop; \
end

module example_tb;
logic clk;
logic rst;

reg [15:0] net_in0, net_in1, net_out0, net_out1;

network net(.clk(clk), .rst(rst), .in0(net_in0), .in1(net_in1), .out0(net_out0), .out1(net_out1));

task test;
input [15:0] in0, in1, out0, out1;
begin
    net_in0 = in0;
    net_in1 = in1;
    #1000ns
    `assert_close(out0, net_out0, 100);
    `assert_close(out1, net_out1, 100);
end
endtask

initial
begin
    test(0, 0, 500, 578);
    test(1000, 1000, 708, 612);
    test(0, 1000, 589, 597);
    test(1000, 0, 619, 595);
    test(1000, 1000, 708, 612);
    test(0, 0, 500, 578);
    test(0, 0, 500, 578);
    test(0, 1000, 589, 597);
    test(0, 1000, 589, 597);
    test(0, 0, 500, 578);
    $display("SUCCESS!");
end
endmodule
