#!/usr/bin/python

import layer_generator


def generate_network_code(num_inputs, layers_size, bus_width):
    num_outputs = layers_size[-1]
    input_wires = ['in{}'.format(i) for i in range(num_inputs)]
    output_wires = ['out{}'.format(i) for i in range(num_outputs)]

    # module declaration
    code = 'module network(clk, rst, '
    for i in range(num_inputs):
        code += input_wires[i] + ', '
    for i in range(num_outputs - 1):
        code += output_wires[i] + ', '
    code += output_wires[num_outputs - 1] + ');\n'
    code += '\n'

    # ports definition
    code += 'input wire clk;\n' + 'input wire rst;\n'
    code += '\n'

    for i in range(num_inputs):
        code += 'input [{}:0] {};\n'.format(bus_width - 1, input_wires[i])
    code += '\n'

    for i in range(num_outputs):
        code += 'output [{}:0] {};\n'.format(bus_width - 1, output_wires[i])
    code += '\n'

    # connectors
    for i in range(len(layers_size) - 1):
        code += 'wire[{}:0] con{}[0:{}];\n'.format(bus_width - 1, i, layers_size[i])
    code += '\n'

    # input layer
    code += 'layer{}in{}out layer0(.clk(clk), .rst(rst), '.format(num_inputs, layers_size[0])
    for i in range(num_inputs):
        code += '.in{}({}), '.format(i, input_wires[i])
    for i in range(layers_size[0] - 1):
        code += '.out{}(con0[{}]), '.format(i, i)
    code += '.out{}(con0[{}]));\n'.format(layers_size[0] - 1, layers_size[0] - 1)

    # hidden layers
    for i in range(1, len(layers_size) - 1):
        code += 'layer{}in{}out layer{}(.clk(clk), .rst(rst), '.format(layers_size[i - 1], layers_size[i], i)
        for j in range(layers_size[i - 1]):
            code += '.in{}(con{}[{}]), '.format(j, i - 1, j)
        for j in range(layers_size[i] - 1):
            code += '.out{}(con{}[{}]), '.format(j, i, j)
        code += '.out{}(con{}[{}]));\n'.format(layers_size[i] - 1, i, layers_size[i] - 1)

    # output layer
    i = len(layers_size) - 1
    code += 'layer{}in{}out layer{}(.clk(clk), .rst(rst), '.format(layers_size[-2], num_outputs, i)
    for j in range(layers_size[i - 1]):
        code += '.in{}(con{}[{}]), '.format(j, i - 1, j)
    for j in range(num_outputs - 1):
        code += '.out{}({}), '.format(j, output_wires[j])
    code += '.out{}({}));\n'.format(num_outputs - 1, output_wires[-1])
    code += '\n'

    code += 'endmodule'
    return code


print(generate_network_code(2, [2, 2], 8))
