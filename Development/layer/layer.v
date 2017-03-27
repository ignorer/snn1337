module layer5(clk, rst, in0, in1, in2, in3, out0, out1, out2, out3, out4);

input wire clk;
input wire rst;

input [7:0] in0;
input [7:0] in1;
input [7:0] in2;
input [7:0] in3;

output [7:0] out0;
output [7:0] out1;
output [7:0] out2;
output [7:0] out3;
output [7:0] out4;

neuron neuron0(.clk(clk), .rst(rst), .in0(in0), .in1(in1), .in2(in2), .in3(in3), .out(out0));
neuron neuron1(.clk(clk), .rst(rst), .in0(in0), .in1(in1), .in2(in2), .in3(in3), .out(out1));
neuron neuron2(.clk(clk), .rst(rst), .in0(in0), .in1(in1), .in2(in2), .in3(in3), .out(out2));
neuron neuron3(.clk(clk), .rst(rst), .in0(in0), .in1(in1), .in2(in2), .in3(in3), .out(out3));
neuron neuron4(.clk(clk), .rst(rst), .in0(in0), .in1(in1), .in2(in2), .in3(in3), .out(out4));

endmodule
