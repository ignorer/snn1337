class FCNNGenerator:
    def __init__(self, bus_width=8):
        self.bus_width = bus_width

    def generate_layer_module(self, num_inputs, num_outputs):
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
            source += f'input [{self.bus_width - 1}:0] in{i};\n'
        source += '\n'

        for i in range(num_outputs):
            source += f'output [{self.bus_width - 1}:0] out{i};\n'
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

    @staticmethod
    def generate_parameter_pass(weight_matrix):
        source = '#('
        for x in range(weight_matrix.shape[0]):
            for y in range(weight_matrix.shape[1]):
                if x == weight_matrix.shape[0] - 1 and y == weight_matrix.shape[1] - 1:
                    source += f'.W{x}TO{y}({weight_matrix[x, y]})) '
                else:
                    source += f'.W{x}TO{y}({weight_matrix[x, y]}), '
        return source

    def generate_network_module(self, weight_matrices):
        layers_size = [weight_matrices[i].shape[1] for i in range(len(weight_matrices))]
        num_inputs = weight_matrices[0].shape[0]
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
            source += f'input [{self.bus_width - 1}:0] in{i};\n'
        source += '\n'

        for i in range(num_outputs):
            source += f'output [{self.bus_width - 1}:0] out{i};\n'
        source += '\n'

        # connectors
        for i in range(len(layers_size) - 1):
            source += f'wire[{self.bus_width - 1}:0] con{i}[0:{layers_size[i] - 1}];\n'
        source += '\n'

        # input layer
        source += f'layer{num_inputs}in{layers_size[0]}out '

        source += self.generate_parameter_pass(weight_matrices[0])
        source += 'layer0(.clk(clk), .rst(rst), '
        for i in range(num_inputs):
            source += f'.in{i}(in{i}), '
        for i in range(layers_size[0] - 1):
            source += f'.out{i}(con0[{i}]), '
        source += f'.out{layers_size[0] - 1}(con0[{layers_size[0] - 1}]));\n'

        # hidden layers
        for i in range(1, len(layers_size) - 1):
            source += f'layer{layers_size[i - 1]}in{layers_size[i]}out '
            source += self.generate_parameter_pass(weight_matrices[i])
            source += f'layer{i}(.clk(clk), .rst(rst), '
            for j in range(layers_size[i - 1]):
                source += f'.in{j}(con{i - 1}[{j}]), '
            for j in range(layers_size[i] - 1):
                source += f'.out{j}(con{i}[{j}]), '
            source += f'.out{layers_size[i] - 1}(con{i}[{layers_size[i] - 1}]));\n'

        # output layer
        i = len(layers_size) - 1
        source += f'layer{layers_size[i - 1]}in{layers_size[i]}out '
        source += self.generate_parameter_pass(weight_matrices[-1])
        source += f'layer{i}(.clk(clk), .rst(rst), '
        for j in range(layers_size[i - 1]):
            source += f'.in{j}(con{i - 1}[{j}]), '
        for j in range(num_outputs - 1):
            source += f'.out{j}(.out{j}), '
        source += f'.out{num_outputs - 1}(out{num_outputs - 1}));\n\n'

        source += 'endmodule\n'
        return source

    def generate_neuron_module(self, num_inputs, extended_width=32):
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
            source += f'input [{self.bus_width - 1}:0] in{i};\n'
        source += '\n'

        source += f'output reg [{self.bus_width - 1}:0] out;\n\n'

        # neuron logic
        source += f'reg signed [{extended_width - 1}:0] x;\n'
        source += f'reg [{extended_width - 1}:0] abs_x;\n'
        source += f'reg [{extended_width - 1}:0] y;\n'
        source += 'always @* begin\n'
        source += '    x = '
        for i in range(num_inputs - 1):
            source += f'in{i} * W{i} + '
        source += f'in{num_inputs - 1} * W{num_inputs - 1};\n'
        source += '    abs_x = x < 0 ? -x : x;\n'
        source += '    if (abs_x >= 5000) y = 1000;\n'
        source += '    else if (abs_x >= 2375 && abs_x < 5000) y = 31 * abs_x + 843;\n'
        source += '    else if (abs_x >= 1000 && abs_x < 2375) y = 125 * abs_x + 625;\n'
        source += '    else if (abs_x >= 0 && abs_x < 1000) y = 250 * abs_x + 500;\n'
        source += '    out = y / 1000;\n'
        source += 'end\n\n'

        source += 'endmodule\n'

        return source
