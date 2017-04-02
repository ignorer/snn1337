#!/usr/bin/python


# num_outputs is actually number of neurons in the layer
def generate_layer_code(bus_width, num_inputs, num_outputs):
    # module declaration
    source = f'module layer{num_inputs}in{num_outputs}out(clk, rst, '
    for i in range(num_inputs):
        source += f'in{i}, '
    for i in range(num_outputs - 1):
        source += f'out{i}, '
    source += f'out{num_outputs - 1});\n\n'

    # parameters
    for i in range(num_inputs):
        for j in range(num_outputs):
            source += f'parameter W{i}TO{j} = 0;\n'
    source += '\n'

    # ports definition
    source += 'input wire clk;\ninput wire rst;\n\n'

    for i in range(num_inputs):
        source += f'input [{bus_width - 1}:0] in{i};\n'
    source += '\n'

    for i in range(num_outputs):
        source += f'output [{bus_width - 1}:0] out{i};\n'
    source += '\n'

    # neurons
    for i in range(num_outputs):
        source += f'neuron{num_inputs}in #('
        for j in range(num_inputs - 1):
            source += f'.W{j}(W{j}TO{i}), '
        source += f'.W{num_inputs - 1}(W{num_inputs - 1}TO{i})) '
        source += f'neuron{i}(.clk(clk), .rst(rst), '
        for j in range(num_inputs):
            source += f'.in{j}(in{j}), '
        source += f'.out(out{i}));\n'
    source += '\n'

    source += 'endmodule\n'
    return source


print(generate_layer_code(bus_width=8, num_inputs=2, num_outputs=2))
# print(generate_layer_code(bus_width=8, weight_matrix=np.matrix([[2, 2, 3], [3, 2, 2]])))
