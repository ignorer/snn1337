#!/usr/bin/python


# num_outputs is actually number of neurons in the layer
def generate_layer_code(bus_width, num_inputs, num_outputs, weight_vector):

    # module declaration
    code = f'module layer{num_inputs}in{num_outputs}out(clk, rst, '
    for i in range(num_inputs):
        code += f'in{i}, '
    for i in range(num_outputs - 1):
        code += f'out{i}, '
    code += f'out{num_outputs - 1});\n\n'
    'in{i}'
    'out{i}'
    # parameters
    for i in range(len(weight_vector)):
        code += f'parameter W{i};\n'
    code += '\n'

    # ports definition
    code += 'input wire clk;\ninput wire rst;\n\n'

    for i in range(num_inputs):
        code += f'input [{bus_width - 1}:0] in{i};\n'
    code += '\n'

    for i in range(num_outputs):
        code += f'output [{bus_width - 1}:0] out{i};\n'
    code += '\n'

    # neurons
    for i in range(num_outputs):
        code += f'neuron{num_inputs}in #('
        for j in range(num_inputs - 1):
            code += f'.W{j}(W{j}), '
        code += f'.W{num_inputs - 1}(W{num_inputs - 1})) '
        code += f'neuron{i}(.clk(clk), .rst(rst), '
        for j in range(num_inputs):
            code += f'.in{j}(in{j}), '
        code += f'.out(out{i}));\n'
    code += '\n'

    code += 'endmodule\n'
    return code

print(generate_layer_code(bus_width=8, num_inputs=2, num_outputs=2, weight_vector=[2, 2]))