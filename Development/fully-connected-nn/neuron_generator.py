#!/usr/bin/python


def neuron_generator(bus_width, num_inputs):

    # module declaration
    source = f'module neuron{num_inputs}in(clk, rst, '
    for i in range(num_inputs):
        source += f'in{i}, '
    source += 'out);\n\n'

    # parameters
    source += 'parameter EXP = 2.72;\n'
    for i in range(num_inputs):
        source += f'parameter W{i} = 0;\n'
    source += '\n'

    # ports definition
    source += 'input wire clk;\ninput wire rst;\n\n'

    for i in range(num_inputs):
        source += f'input [{bus_width - 1}:0] in{i};\n'
    source += '\n'

    source += f'output [{bus_width - 1}:0] out;\n\n'

    # neuron logic
    sum_source = ''
    for i in range(num_inputs - 1):
        sum_source += f'in{i} * W{i} + '
    sum_source += f'in{num_inputs - 1} * W{num_inputs - 1}'
    source += f'assign out = 1 / (1 + EXP ** -({sum_source}));\n\n'

    source += 'endmodule\n'

    return source


print(neuron_generator(bus_width=8, num_inputs=4))
