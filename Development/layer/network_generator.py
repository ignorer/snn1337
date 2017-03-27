#!/usr/bin/python

# import layer_generator


def generate_network_code(num_inputs, num_outputs, layers_size, bus_width):
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

    # TODO

    code += 'endmodule'
    return code


print(generate_network_code(2, 1, [2, 2], 8))
