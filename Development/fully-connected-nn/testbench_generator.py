import fcnn_generator as code_generator
import simple_fcnn as network_generator
import numpy as np

class FCNNTestbenchGenerator():
    def __init__(self, bus_width=16, decimal_precision=3, eps = 0.1, seed=1337):
        self.bus_width = bus_width
        self.decimal_precision = decimal_precision
        self.eps = eps
        self.seed = seed

    def generate_testbench(self, layer_sizes, test_count):
        fcnn = network_generator.FCNN(layer_sizes, self.seed, network_generator.fpga_sigmoid)
        fpgaNet = fcnn.get_fpga_network(self.decimal_precision)

        inputs = [np.random.randint(0, 2, layer_sizes[0]).astype(float) for i in range(0, test_count)]
        results = [fcnn.predict(input) for input in inputs]

        codeGen = code_generator.FCNNGenerator(self.bus_width, self.decimal_precision)

        # start generating code
        source = codeGen.generate_network_module(fpgaNet)

        # assert
        source += '\n`define assert_close(expected, got, eps) \\\n'
        source += 'if ((expected > got && expected > got + eps) || (expected < got && expected + eps < got)) begin \\\n'
        source += '    $display("TEST FAILED in %m: got %d, exprected %d", got, expected); \\\n'
        source += '    $stop; \\\n'
        source += 'end\n\n'

        # testbench
        source += 'module example_tb;\n'
        source += 'logic clk;\n'
        source += 'logic rst;\n'

        # inputs/outputs
        source += f'\nreg [{self.bus_width - 1}:0] '
        for i in range(0, layer_sizes[0]):
            source += f'net_in{i}, '
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
        source += f'input [{self.bus_width - 1}:0] '
        for i in range(0, layer_sizes[0]):
            source += f'in{i}, '
        for i in range(0, layer_sizes[-1] - 1):
            source += f'out{i}, '
        source += f'out{layer_sizes[-1] - 1};\n'
        source += 'begin\n'
        for i in range(0, layer_sizes[0]):
            source += f'    net_in{i} = in{i};\n'
        source += '    #1000ns\n'

        for i in range(0, layer_sizes[-1]):
            source += f'    `assert_close(out{i}, net_out{i}, {int(self.eps * (10 ** self.decimal_precision))});\n'
        source += 'end\n'
        source += 'endtask\n'

        # run all tests
        source += '\ninitial\n'
        source += 'begin\n'
        for i in range(0, test_count):
            source += '    test('
            for j in range(0, layer_sizes[0]):
                source += f'{int(inputs[i][j] * (10 ** self.decimal_precision))}, '
            for j in range(0, layer_sizes[-1] - 1):
                source += f'{int(results[i][j][0] * (10 ** self.decimal_precision) + 0.5)}, '
            source += f'{int(results[i][layer_sizes[-1] - 1][0] * (10 ** self.decimal_precision) + 0.5)});\n'
        source += '    $display("SUCCESS!");\n'
        source += 'end\n'
        source += 'endmodule\n'
        return source

if (__name__ == '__main__'):
    t = FCNNTestbenchGenerator()
    print(t.generate_testbench([2, 1], 10))
