#!/usr/bin/python


def generate_neuron_code(bus_width, num_inputs):

    # module declaration
    source = f'module neuron{num_inputs}in(clk, rst, '
    for i in range(num_inputs):
        source += f'in{i}, '
    source += 'out);\n\n'

    # parameters
    for i in range(num_inputs):
        source += f'parameter W{i} = 0;\n'
    source += '\n'

    # ports definition
    source += 'input wire clk;\ninput wire rst;\n\n'

    for i in range(num_inputs):
        source += f'input [{bus_width - 1}:0] in{i};\n'
    source += '\n'

    source += f'output reg [{bus_width - 1}:0] out;\n\n'

    # neuron logic
    extended_width = 32
    source += f'reg signed [{extended_width - 1}:0] x;\n'
    source += f'reg [{extended_width - 1}:0] abs_x;\n'
    source += 'always @* begin\n'
    source += '    x = '
    for i in range(num_inputs - 1):
        source += f'in{i} * W{i} + '
    source += f'in{num_inputs - 1} * W{num_inputs - 1};\n'
    source += '    abs_x = x < 0 ? -x : x;\n'
    source += '    if (abs_x >= 5000) out = 1000;\n'
    source += '    else if (abs_x >= 2375 && abs_x < 5000) out = 31 * abs_x + 843;\n'
    source += '    else if (abs_x >= 1000 && abs_x < 2375) out = 125 * abs_x + 625;\n'
    source += '    else if (abs_x >= 0 && abs_x < 1000) out = 250 * abs_x + 500;\n'
    source += 'end\n\n'

    source += 'endmodule\n'

    return source


print(generate_neuron_code(bus_width=8, num_inputs=16))
