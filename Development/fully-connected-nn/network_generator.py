#!/usr/bin/python

import numpy as np


def generate_parameter_pass(weight_matrix):
    source = '#('
    for x in range(weight_matrix.shape[0]):
        for y in range(weight_matrix.shape[1]):
            if x == weight_matrix.shape[0] - 1 and y == weight_matrix.shape[1] - 1:
                source += f'.W{x}TO{y}({weight_matrix[x, y]})) '
            else:
                source += f'.W{x}TO{y}({weight_matrix[x, y]}), '
    return source


def generate_network_code(bus_width, weight_matrices):
    layers_size = [weight_matrices[i].shape[1] for i in range(len(weight_matrices))]
    num_inputs = weight_matrices[0].shape[1]
    num_outputs = layers_size[-1]

    # module declaration
    source = 'module network(clk, rst, '
    for i in range(num_inputs):
        source += f'in{i}, '
    for i in range(num_outputs - 1):
        source += f'out{i}, '
    source += f'out{num_outputs - 1});\n\n'

    # ports definition
    source += 'input wire clk;\n' + 'input wire rst;\n\n'

    for i in range(num_inputs):
        source += f'input [{bus_width - 1}:0] in{i};\n'
    source += '\n'

    for i in range(num_outputs):
        source += f'output [{bus_width - 1}:0] out{i};\n'
    source += '\n'

    # connectors
    for i in range(len(layers_size) - 1):
        source += f'wire[{bus_width - 1}:0] con{i}[0:{layers_size[i] - 1}];\n'
    source += '\n'

    # input layer
    source += f'layer{num_inputs}in{layers_size[0]}out '

    source += generate_parameter_pass(weight_matrices[0])
    source += 'layer0(.clk(clk), .rst(rst), '
    for i in range(num_inputs):
        source += f'.in{i}(in{i}), '
    for i in range(layers_size[0] - 1):
        source += f'.out{i}(con0[{i}]), '
    source += f'.out{layers_size[0] - 1}(con0[{layers_size[0] - 1}]));\n'

    # hidden layers
    for i in range(1, len(layers_size) - 1):
        source += f'layer{layers_size[i - 1]}in{layers_size[i]}out '
        source += generate_parameter_pass(weight_matrices[i])
        source += f'layer{i}(.clk(clk), .rst(rst), '
        for j in range(layers_size[i - 1]):
            source += f'.in{j}(con{i - 1}[{j}]), '
        for j in range(layers_size[i] - 1):
            source += f'.out{j}(con{i}[{j}]), '
        source += f'.out{layers_size[i] - 1}(con{i}[{layers_size[i] - 1}]));\n'

    # output layer
    i = len(layers_size) - 1
    source += f'layer{layers_size[i - 1]}in{layers_size[i]}out '
    source += generate_parameter_pass(weight_matrices[-1])
    source += f'layer{i}(.clk(clk), .rst(rst), '
    for j in range(layers_size[i - 1]):
        source += f'.in{j}(con{i - 1}[{j}]), '
    for j in range(num_outputs - 1):
        source += f'.out{j}(.out{j}), '
    source += f'.out{num_outputs - 1}(out{num_outputs - 1}));\n\n'

    source += 'endmodule\n'
    return source


print(generate_network_code(bus_width=8, weight_matrices=[np.matrix([[1, 2], [3, 4]]),
                                                          np.matrix([[1, 2, 3], [5, 6, 7]]),
                                                          np.matrix([[7], [8], [3]])]))
