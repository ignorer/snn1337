#!/usr/bin/python


def neuron_generator(num_inputs):
    input_wires = ['in{}'.format(i) for i in range(num_inputs)]

    # module declaration
    code = f'module neuron{num_inputs}in(clk, rst, '
    for i in range(num_inputs):
        code += f'in{i}, '
    code += 'out);\n\n'

    # parameters
    for i in range(num_inputs):
        code += f'parameter W{i};\n'
    code += '\n'

    code += 'endmodule\n'

    return code


print(neuron_generator(num_inputs=2))
