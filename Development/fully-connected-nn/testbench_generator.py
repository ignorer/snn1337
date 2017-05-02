import fcnn_generator as code_generator
import simple_fcnn as network_generator
import numpy as np


class FCNNTestbenchGenerator:
    def __init__(self, network, bus_width=20, decimal_precision=3, eps=0.01, seed=1337):
        self.network = network
        self.bus_width = bus_width
        self.decimal_precision = decimal_precision
        self.eps = eps
        self.seed = seed

    def generate_testbench(self, inputs):
        layer_sizes = self.network.get_layer_sizes()
        weights = self.network.get_fpga_weights(self.decimal_precision)
        bias = self.network.get_fpga_bias(self.decimal_precision)

        outputs = [self.network.predict(input)[-1] for input in inputs]

        code_gen = code_generator.FCNNGenerator(self.bus_width, self.decimal_precision)

        # start generating code
        source = code_gen.generate_network_module(weights, bias)

        # assert
        source += '\n`define assert_close(expected, got, eps) \\\n'
        source += 'if ((expected > got && expected > got + eps) || (expected < got && expected + eps < got)) begin \\\n'
        source += '    $display("TEST FAILED in %m: got %d, expected %d", got, expected); \\\n'
        source += 'end\n\n'

        # testbench
        source += 'module example_tb;\n'
        source += 'logic clk;\n'
        source += 'logic rst;\n'

        # inputs/outputs
        source += f'\nreg signed [{self.bus_width - 1}:0] '
        for i in range(0, layer_sizes[0] - 1):
            source += f'net_in{i}, '
        source += f'net_in{layer_sizes[0] - 1};\n'
        source += f'\nwire signed [{self.bus_width - 1}:0] '
        for i in range(0, layer_sizes[-1] - 1):
            source += f'net_out{i}, '
        source += f'net_out{layer_sizes[-1] - 1};\n'

        # network
        source += '\nnetwork net(.clk(clk), .rst(rst), '
        for i in range(0, layer_sizes[0]):
            source += f'.in{i}(net_in{i}), '
        for i in range(0, layer_sizes[-1] - 1):
            source += f'.out{i}(net_out{i}), '
        source += f'.out{layer_sizes[-1] - 1}(net_out{layer_sizes[-1] - 1}));\n'

        # test
        source += '\ntask test;\n'
        source += f'input signed [{self.bus_width - 1}:0] '
        for i in range(0, layer_sizes[0]):
            source += f'in{i}, '
        for i in range(0, layer_sizes[-1] - 1):
            source += f'out{i}, '
        source += f'out{layer_sizes[-1] - 1};\n'
        source += 'begin\n'
        for i in range(0, layer_sizes[0]):
            source += f'    net_in{i} <= in{i};\n'
        source += '    #10000ns\n'

        for i in range(0, layer_sizes[-1]):
            source += f'    `assert_close(out{i}, net_out{i}, {int(self.eps * (10 ** self.decimal_precision))});\n'
        source += 'end\n'
        source += 'endtask\n'

        # run all tests
        source += '\ninitial\n'
        source += 'begin\n'
        source += '    $dumpfile("waves.vcd");\n'
        source += '    $dumpvars;\n'
        for i in range(0, len(outputs)):
            source += '    test('
            for j in range(0, layer_sizes[0]):
                source += f'{int(inputs[i][j] * (10 ** self.decimal_precision))}, '
            for j in range(0, layer_sizes[-1] - 1):
                source += f'{int(outputs[i][j] * (10 ** self.decimal_precision) + 0.5)}, '
            source += f'{int(outputs[i][layer_sizes[-1] - 1] * (10 ** self.decimal_precision) + 0.5)});\n'
            source += f'    $display("Test{i} completed");\n'
        source += '    $display("SUCCESS!");\n'
        source += 'end\n'
        source += 'endmodule\n'
        return source


if __name__ == '__main__':
    fcnn = network_generator.get_random_network([65, 64, 1])
    generator = FCNNTestbenchGenerator(fcnn, 16, 3)
    test_count = 100
    inputs = [np.random.randint(0, 2, 65).astype(float) for _ in range(0, test_count)]
    output = generator.generate_testbench(inputs)
    with open('fcnn_testbench.sv', 'w') as output_file:
        output_file.write(output)
    print(output)
