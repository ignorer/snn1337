def generate_layer_code(num_neurons, bus_width):
    input_wires = ['in{}'.format(i) for i in range(num_neurons)]
    output_wires = ['out{}'.format(i) for i in range(num_neurons)]

    # module declaration
    code = 'module layer{}(clk, rst, '.format(num_neurons)
    for i in range(num_neurons):
        code += input_wires[i] + ', '
    for i in range(num_neurons - 1):
        code += output_wires[i] + ', '
    code += output_wires[num_neurons - 1] + ');\n'
    code += '\n'

    # ports definition
    code += 'input wire clk;\n' + 'input wire rst;\n'
    code += '\n'

    for i in range(num_neurons):
        code += 'input [{}:0] {};\n'.format(bus_width - 1, input_wires[i])
    code += '\n'

    for i in range(num_neurons):
        code += 'output [{}:0] {};\n'.format(bus_width - 1, output_wires[i])
    code += '\n'

    # neurons
    for i in range(num_neurons):
        code += 'neuron neuron{}'.format(i) + '('
        code += '.clk(clk), .rst(rst), '
        for j in range(num_neurons):
            code += '.in{}({}), '.format(j, input_wires[j])
        code += '.out(out{}));\n'.format(i)
    code += '\n'

    code += 'endmodule\n'
    return code

print(generate_layer_code(5, 8))
