#!/usr/bin/python


def generate_network_code(bus_width, num_inputs, layers_size, weight_matrix):
    num_outputs = layers_size[-1]

    # module declaration
    code = 'module network(clk, rst, '
    for i in range(num_inputs):
        code += f'in{i}, '
    for i in range(num_outputs - 1):
        code += f'out{i}, '
    code += f'out{num_outputs - 1});\n\n'

    # ports definition
    code += 'input wire clk;\n' + 'input wire rst;\n\n'

    for i in range(num_inputs):
        code += f'input [{bus_width - 1}:0] in{i};\n'
    code += '\n'

    for i in range(num_outputs):
        code += f'output [{bus_width - 1}:0] out{i};\n'
    code += '\n'

    # connectors
    for i in range(len(layers_size) - 1):
        code += f'wire[{bus_width - 1}:0] con{i}[0:{layers_size[i]}];\n'
    code += '\n'

    # input layer
    code += f'layer{num_inputs}in{layers_size[0]}out #('
    for i in range(len(weight_matrix[0]) - 1):
        code += f'.W{i}({weight_matrix[0][i]}), '
    code += f'.W{len(weight_matrix[0]) - 1}({weight_matrix[0][-1]})) '
    code += 'layer0(.clk(clk), .rst(rst), '
    for i in range(num_inputs):
        code += f'.in{i}(in{i}), '
    for i in range(layers_size[0] - 1):
        code += f'.out{i}(con0[{i}]), '
    code += f'.out{layers_size[0] - 1}(con0[{layers_size[0] - 1}]));\n'

    # hidden layers
    for i in range(1, len(layers_size) - 1):
        code += f'layer{layers_size[i - 1]}in{layers_size[i]}out #('
        for j in range(len(weight_matrix[i]) - 1):
            code += f'.W{j}({weight_matrix[i][j]}), '
        code += f'.W{len(weight_matrix[i]) - 1}({weight_matrix[i][-1]})) '
        code += f'layer{i}(.clk(clk), .rst(rst), '
        for j in range(layers_size[i - 1]):
            code += f'.in{j}(con{i - 1}[{j}]), '
        for j in range(layers_size[i] - 1):
            code += f'.out{j}(con{i}[{j}]), '
        code += f'.out{layers_size[i] - 1}(con{i}[{layers_size[i] - 1}]));\n'

    # output layer
    i = len(layers_size) - 1
    code += f'layer{layers_size[i - 1]}in{layers_size[i]}out #('
    for j in range(len(weight_matrix[i]) - 1):
        code += f'.W{j}({weight_matrix[i][j]}), '
    code += f'.W{len(weight_matrix[i]) - 1}({weight_matrix[i][-1]})) '
    code += f'layer{i}(.clk(clk), .rst(rst), '
    for j in range(layers_size[i - 1]):
        code += f'.in{j}(con{i - 1}[{j}]), '
    for j in range(num_outputs - 1):
        code += f'.out{j}(.out{j}), '
    code += f'.out{num_outputs - 1}(out{num_outputs - 1}));\n\n'

    code += 'endmodule\n'
    return code


print(generate_network_code(bus_width=8, num_inputs=2, layers_size=[2, 2, 1],
                            weight_matrix=[[1, 2], [3, 4], [5]]))
