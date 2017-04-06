import numpy as np

class FCNNGenerator:
    def __init__(self, bus_width=16, decimal_precision=3):
        self.bus_width = bus_width
        self.extended_width = bus_width * 2
        self.decimal_precision = decimal_precision
        self.neurons = dict()
        self.layers = dict()

    def generate_layer_module(self, num_inputs, num_outputs):
        if ((num_inputs, num_outputs) in self.layers):
            return self.layers[(num_inputs, num_outputs)]
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

        self.generate_neuron_module(num_inputs)
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

        source += 'endmodule\n\n'
        self.layers[(num_inputs, num_outputs)] = source
        return self.layers[(num_inputs, num_outputs)]

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
        self.generate_layer_module(num_inputs, layers_size[0])
        source += f'layer{num_inputs}in{layers_size[0]}out '

        source += self.generate_parameter_pass(weight_matrices[0])
        source += 'layer0(.clk(clk), .rst(rst), '
        for i in range(num_inputs):
            source += f'.in{i}(in{i}), '
        for i in range(layers_size[0] - 1):
            source += f'.out{i}('
            if len(layers_size) > 1:
                source += f'con0[{i}]'
            else:
                source += f'out{i}'
            source += '), '
        source += f'.out{layers_size[0] - 1}('
        if len(layers_size) > 1:
            source += f'con0[{layers_size[0] - 1}]'
        else:
            source += f'out{layers_size[0] - 1}'
        source += '));\n'

        # hidden layers
        for i in range(1, len(layers_size) - 1):
            self.generate_layer_module(layers_size[i - 1], layers_size[i])
            source += f'layer{layers_size[i - 1]}in{layers_size[i]}out '
            source += self.generate_parameter_pass(weight_matrices[i])
            source += f'layer{i}(.clk(clk), .rst(rst), '
            for j in range(layers_size[i - 1]):
                source += f'.in{j}(con{i - 1}[{j}]), '
            for j in range(layers_size[i] - 1):
                source += f'.out{j}(con{i}[{j}]), '
            source += f'.out{layers_size[i] - 1}(con{i}[{layers_size[i] - 1}]));\n'

        # output layer
        if len(layers_size) > 1:
            i = len(layers_size) - 1
            self.generate_layer_module(layers_size[i - 1], layers_size[i])
            source += f'layer{layers_size[i - 1]}in{layers_size[i]}out '
            source += self.generate_parameter_pass(weight_matrices[-1])
            source += f'layer{i}(.clk(clk), .rst(rst), '
            for j in range(layers_size[i - 1]):
                source += f'.in{j}(con{i - 1}[{j}]), '
            for j in range(num_outputs - 1):
                source += f'.out{j}(out{j}), '
            source += f'.out{num_outputs - 1}(out{num_outputs - 1}));\n\n'

        source += 'endmodule\n'

        depends = ""
        for neuron in self.neurons.values():
            depends += neuron
        for layer in self.layers.values():
            depends += layer
        return depends + source

    def get_decimal(self, num):
        return int(num * (10 ** self.decimal_precision) + 0.5)

    def generate_neuron_module(self, num_inputs):
        if (num_inputs in self.neurons):
            return self.neurons[num_inputs]
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
            source += f'input signed [{self.bus_width - 1}:0] in{i};\n'
        source += '\n'

        source += f'output reg [{self.bus_width - 1}:0] out;\n\n'

        # neuron logic
        source += f'reg signed [{self.extended_width - 1}:0] x;\n'
        source += f'reg [{self.extended_width - 1}:0] abs_x;\n'
        source += f'reg [{self.extended_width - 1}:0] y;\n'
        source += 'always @* begin\n'
        source += '    x = '
        for i in range(num_inputs - 1):
            source += f'in{i} * W{i} / {self.get_decimal(1)} + '
        source += f'in{num_inputs - 1} * W{num_inputs - 1} / {self.get_decimal(1)};\n'
        source += '    abs_x = x < 0 ? -x : x;\n'
        source += f'    if (abs_x >= {self.get_decimal(5)}) y = {self.get_decimal(1)};\n'
        source += f'    else if (abs_x >= {self.get_decimal(2.375)}) y = {self.get_decimal(0.03125)} * abs_x / {self.get_decimal(1)} + {self.get_decimal(0.84375)};\n'
        source += f'    else if (abs_x >= {self.get_decimal(1)}) y = {self.get_decimal(0.125)} * abs_x / {self.get_decimal(1)} + {self.get_decimal(0.625)};\n'
        source += f'    else if (abs_x >= 0) y = {self.get_decimal(0.25)} * abs_x / {self.get_decimal(1)} + {self.get_decimal(0.5)};\n'
        source += '    out = y;\n'
        source += 'end\n\n'
        source += 'endmodule\n\n'
        self.neurons[num_inputs] = source
        return self.neurons[num_inputs]

if __name__ == '__main__':
    generator = FCNNGenerator()
    weights = [
        np.matrix(np.array([
            1, 2,
            3, 4,
        ]).reshape(2, 2)),
        np.matrix(np.array([
            5, 6,
            7, 8,
        ]).reshape(2, 2)),
        np.matrix(np.array([
            9, 10
        ]).reshape((2, 1)))
    ]
    print(generator.generate_network_module(weights))