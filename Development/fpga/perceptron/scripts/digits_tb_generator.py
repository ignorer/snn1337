import numpy as np
from fcnn_generator import FCNNGenerator


class FCNNDigitsTestbenchGenerator:
    def __init__(self, bus_width=16, decimal_precision=5):
        self.bus_width = bus_width
        self.decimal_precision = decimal_precision
        self.num_inputs = 64
        self.num_outputs = 10

    def read_data(self):
        input_data = np.loadtxt('../test/input_digits.txt', dtype='i', delimiter=' ')
        input_data *= 10 ** self.decimal_precision
        output_data = np.loadtxt('../test/output_digits.txt', dtype='i', delimiter=' ')
        output_data *= 10 ** self.decimal_precision

        num_layers = 2
        weight_matrices = []
        with open('../test/network_digits.txt') as network_data_file:
            for _ in range(num_layers):
                num_lines = int(network_data_file.readline())
                matrix = []
                for j in range(num_lines):
                    matrix.append(list(map(float, network_data_file.readline().strip().split())))
                weight_matrices.append(matrix)

                _ = network_data_file.readline()  # skip bias

        weight_matrices = list(
            map(lambda x: (np.matrix(x) * 10 ** self.decimal_precision).astype(int), weight_matrices))
        return input_data, output_data, weight_matrices

    def generate_testbench(self):
        code_generator = FCNNGenerator(decimal_precision=self.decimal_precision)

        # reading data
        input_sample, expected_output, weight_matrices = self.read_data()

        # generating network modules
        source = code_generator.generate_network_module(weight_matrices)

        # testbench module
        source += '\nmodule testbench_digits;\n\n'
        source += 'logic clk;\nlogic rst;\n\n'

        source += f'reg [{self.bus_width - 1}:0] '
        for i in range(self.num_inputs - 1):
            source += f'net_in{i}, '
        source += f'net_in{self.num_inputs - 1};\n'
        source += f'wire [{self.bus_width - 1}:0] '
        for i in range(self.num_outputs - 1):
            source += f'net_out{i}, '
        source += f'net_out{self.num_outputs - 1};\n\n'

        # network
        source += 'network net(.clk(clk), .rst(rst), '
        for i in range(self.num_inputs):
            source += f'.in{i}(net_in{i}), '
        for i in range(0, self.num_outputs - 1):
            source += f'.out{i}(net_out{i}), '
        source += f'.out{self.num_outputs - 1}(net_out{self.num_outputs - 1}));\n\n'

        # task test
        source += 'task test;\n'
        source += f'input [{self.bus_width - 1}:0] '
        for i in range(self.num_inputs - 1):
            source += f'in{i}, '
        source += f'in{self.num_inputs - 1};\n'
        source += 'begin\n'
        for i in range(self.num_inputs):
            source += f'    net_in{i} = in{i};\n'
        source += '    #10000000\n\n'

        for i in range(self.num_outputs):
            source += f'    $write("%d ", net_out{i});\n'
        source += '    $display();\n'
        source += 'end\n'
        source += 'endtask\n\n'

        # run all tests
        source += 'initial begin\n'
        for i in range(10):  # 10 tests
            source += '    test('
            for j in range(self.num_inputs - 1):
                source += f'{input_sample[i, j]}, '
            source += f'{input_sample[i, self.num_inputs - 1 ]});\n'

        source += 'end\n'
        source += 'endmodule\n'

        return source


if __name__ == '__main__':
    testbench_generator = FCNNDigitsTestbenchGenerator(decimal_precision=3)
    verilog_source = testbench_generator.generate_testbench()
    with open('../test/fcnn_digits_tb.sv', 'w') as verilog_file:
        verilog_file.write(verilog_source)
    print(testbench_generator.read_data()[2])
