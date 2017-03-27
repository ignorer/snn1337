#!/usr/bin/python

import sys


# num_outputs is actually number of neurons in the layer
def generate_layer_code(num_inputs, num_outputs, bus_width):
    input_wires = ['in{}'.format(i) for i in range(num_inputs)]
    output_wires = ['out{}'.format(i) for i in range(num_outputs)]

    # module declaration
    code = 'module layer{}in{}out(clk, rst, '.format(num_inputs, num_outputs)
    for i in range(num_inputs):
        code += input_wires[i] + ', '
    for i in range(num_outputs - 1):
        code += output_wires[i] + ', '
    code += output_wires[num_outputs - 1] + ');\n'
    code += '\n'

    # ports definition
    code += 'input wire clk;\ninput wire rst;\n'
    code += '\n'

    for i in range(num_inputs):
        code += 'input [{}:0] {};\n'.format(bus_width - 1, input_wires[i])
    code += '\n'

    for i in range(num_outputs):
        code += 'output [{}:0] {};\n'.format(bus_width - 1, output_wires[i])
    code += '\n'

    # neurons
    for i in range(num_outputs):
        code += 'neuron{}in neuron{}(.clk(clk), .rst(rst), '.format(num_inputs, i)
        for j in range(num_inputs):
            code += '.in{}({}), '.format(j, input_wires[j])
        code += '.out(out{}));\n'.format(i)
    code += '\n'

    code += 'endmodule\n'
    return code

#print(generate_layer_code(2, 2, 8))

#with open(sys.argv[1], 'w') as verilog_file:
#    verilog_file.write(generate_layer_code(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])))
