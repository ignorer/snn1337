import numpy as np
from enum import Enum, IntEnum

class ActivationFunctionType(IntEnum):
    SIGMOID_APPROXIMATION = 0
    NON_SATURATING = 1

class LayerType(Enum):
    CONVOLUTIONAL = 0
    MAXPOOL = 1
    ACTIVATION = 2
    DENSE = 3
    FULLY_CONNECTED = 4

class Layer:
    def __init__(self, id, type, inDepth, inWidth, inHeight, outDepth, outWidth, outHeight):
        self.id = id
        self.type = type
        self.inDepth = inDepth
        self.inWidth = inWidth
        self.inHeight = inHeight
        self.outDepth = outDepth
        self.outWidth = outWidth
        self.outHeight = outHeight

class ConvolutionalLayer(Layer):
    def __init__(self, id, weights, bias = None):
        super().__init__(id, LayerType.CONVOLUTIONAL, weights.shape[0], weights.shape[1], weights.shape[2], 1, 1, 1)
        self.weights = weights
        self.bias = bias

class MaxpoolLayer(Layer):
    def __init__(self, id, depth, width, height, stepX, stepY):
        super().__init__(id, LayerType.MAXPOOL, depth, width, height, 1, 1, 1)
        self.stepX = stepX
        self.stepY = stepY

class ActivationLayer(Layer):
    def __init__(self, id, depth, width, height, funType):
        super().__init__(id, LayerType.ACTIVATION, depth, width, height, depth, width, height)
        self.funType = funType

class DenseLayer(Layer):
    def __init__(self, id, depth, width, height, layers, stepX, stepY):
        self.filterDepth = layers[0].inDepth
        self.filterWidth = layers[0].inWidth
        self.filterHeight = layers[0].inHeight
        self.stepX = stepX
        self.stepY = stepY
        self.filters = layers
        super().__init__(id, LayerType.DENSE, depth, width, height, len(layers),
                         (width - self.filterWidth + 1) // stepX, (height - self.filterHeight + 1) // stepY)

class FullyConnectedLayer(Layer):
    def __init__(self, id, weights, bias = None):
        super().__init__(id, LayerType.FULLY_CONNECTED, weights.shape[0], weights.shape[1], weights.shape[2],
                         weights.shape[3], weights.shape[4], weights.shape[5])
        self.weights = weights
        self.bias = bias

class FPGANetworkGenerator:
    def __init__(self, busWidth=20, decimalPrecision=3, eps=0.1):
        self.busWidth = busWidth
        self.extendedWidth = busWidth * 2
        self.decimalPrecision = decimalPrecision
        self.eps = eps
        self.modules = {}

    def add_module(self, id, source):
        self.modules[id] = source

    def get_decimal(self, num):
        return int(num * (10 ** self.decimalPrecision) + 0.5)

    @staticmethod
    def generate_params(depth, width, height, prefix, suffix = ''):
        source = ''
        for i in range(depth):
            for j in range(width):
                for k in range(height):
                    source += f'{prefix}_{i}_{j}_{k}{suffix}'
        return source

    @staticmethod
    def generate_params_pass(depth, width, height, z, x, y, nameFrom, nameTo):
        source = ''
        for i in range(depth):
            for j in range(width):
                for k in range(height):
                    source += f', .{nameTo(i, j, k)}({nameFrom(i + z, j + x, k + y)})'
        return source

    def generate_module_header(self, name, inDepth, inWidth, inHeight, outDepth, outWidth, outHeight, reg = ''):
        source = f'module {name}(clk, rst'
        source += self.generate_params(inDepth, inWidth, inHeight, ', in')
        source += self.generate_params(outDepth, outWidth, outHeight, ', out')
        source += ');\n'

        source += 'input wire clk;\n'
        source += 'input wire rst;\n\n'

        source += self.generate_params(inDepth, inWidth, inHeight, f'input signed [{self.busWidth - 1}:0] in', ';\n')
        source += '\n'
        source += self.generate_params(outDepth, outWidth, outHeight, f'output {reg} signed [{self.busWidth - 1}:0] out', ';\n')
        source += '\n'
        return source

    def generate_module_include(self, name, depth, width, height, z, x, y, outDepth, outX, outY):
        source = f'{name}(.clk(clk), .rst(rst)'
        name = lambda x, y, z: f'in_{x}_{y}_{z}'
        source += self.generate_params_pass(depth, width, height, z, x, y, name, name)
        source += f', .out_0_0_0(out_{outDepth}_{outX}_{outY}));\n'
        return source

    def generate_dense_layer(self, id, inDepth, inWidth, inHeight, filterWidth, filterHeight, stepX, stepY, filters):
        outDepth = len(filters)
        outWidth = (inWidth - filterWidth + 1) // stepX
        outHeight = (inHeight - filterHeight + 1) // stepY

        source = self.generate_module_header(f'dense_{id}', inDepth, inWidth, inHeight, outDepth, outWidth, outHeight)

        for i in range(outDepth):
            for j in range(0, outWidth, stepX):
                for k in range(0, outHeight, stepY):
                    filter_name = f'convolutional_{filters[i].id} filter_{i}_{j}_{k}'
                    if filters[i].type == LayerType.MAXPOOL:
                        self.generate_maxpool_layer(filters[i].id, filters[i].inDepth, filters[i].inWidth, filters[i].inHeight)
                        filter_name = f'maxpool_{filters[i].id} filter_{i}_{j}_{k}'
                    else:
                        self.generate_convolutional_layer(filters[i].id, filters[i].weights, filters[i].bias)
                    source += self.generate_module_include(filter_name, inDepth, filterHeight, filterHeight,
                                                           0, j, k, i, j // stepX, k // stepY)

        source += '\nendmodule\n\n'
        self.add_module(id, source)
        return source

    def generate_pool(self, outName, functions, call, step = 0, res = '', dec = ''):
        size = len(functions)

        if (size == 1):
            res += f'    {outName} <= {functions[0]};\n'
            return [res, dec]

        newFunctions = []
        for i in range(0, size - 1, 2):
            dec += f'reg signed [{self.busWidth - 1}:0] fun_{step}_{i >> 1};\n'
            out = call(functions[i], functions[i + 1])
            res += f'    fun_{step}_{i >> 1} <= '
            if (size % 2 != 0 and i + 3 == size):
                res += f'{call(out, functions[i + 2])};\n'
            else:
                res += f'{out};\n'
            newFunctions.append(f'fun_{step}_{i >> 1}')

        return self.generate_pool(outName, newFunctions, call, step + 1, res, dec)

    # weights with shape (depth, weight, height)
    def generate_convolutional_layer(self, id, weights, bias = None):
        depth = weights.shape[0]
        width = weights.shape[1]
        height = weights.shape[2]

        source = self.generate_module_header(f'convolutional_{id}', depth, width, height, 1, 1, 1, 'reg')

        for i in range(depth):
            for j in range(width):
                for k in range(height):
                    source += f'parameter signed W_{i}_{j}_{k} = {self.get_decimal(weights[i][j][k])};\n'
            if bias is not None:
                source += f'parameter signed BIAS_{i} = {self.get_decimal(bias[i])};\n'

        functions = []
        for i in range(depth):
            for j in range(width):
                for k in range(height):
                    functions.append(f'W_{i}_{j}_{k} * in_{i}_{j}_{k} / {self.get_decimal(1.0)}')
            if bias is not None:
                functions.append(f'BIAS_{i}')

        tmp = self.generate_pool('out_0_0_0', functions, lambda x, y: f'({x} + {y})')
        source += tmp[1]
        source += 'always @* begin\n'
        source += tmp[0]
        source += 'end\nendmodule\n\n'
        self.add_module(id, source)
        return source

    def generate_maxpool_layer(self, id, depth, width, height):
        source = self.generate_module_header(f'maxpool_{id}', depth, width, height, 1, 1, 1, 'reg')

        functions = []
        for i in range(depth):
            for j in range(width):
                for k in range(height):
                    functions.append(f'in_{i}_{j}_{k}')

        tmp = self.generate_pool('out_0_0_0', functions, lambda x, y: f'({x} > {y} ? {x} : {y})')
        source += tmp[1]
        source += 'always @* begin\n'
        source += tmp[0]
        source += 'end\nendmodule\n\n'
        self.add_module(id, source)
        return source

    def generate_sigmoid_approximation(self):
        source =  f'reg signed [{self.extendedWidth - 1}:0] abs_x;\n'
        source += f'reg signed [{self.extendedWidth - 1}:0] y;\n'
        source += f'always @* begin\n'
        source += f'    abs_x = in_0_0_0 < 0 ? -in_0_0_0 : in_0_0_0;\n'
        source += f'    if (abs_x >= {self.get_decimal(5)}) y = {self.get_decimal(1)};\n'
        source += f'    else if (abs_x >= {self.get_decimal(2.375)}) y = {self.get_decimal(0.03125)} * abs_x /' \
                  f' {self.get_decimal(1)} + {self.get_decimal(0.84375)};\n'
        source += f'    else if (abs_x >= {self.get_decimal(1)}) y = {self.get_decimal(0.125)} * abs_x /' \
                  f' {self.get_decimal(1)} + {self.get_decimal(0.625)};\n'
        source += f'    else y = {self.get_decimal(0.25)} * abs_x /' \
                  f' {self.get_decimal(1)} + {self.get_decimal(0.5)};\n'
        source += f'    out_0_0_0 = in_0_0_0 < 0 ? {self.get_decimal(1)} - y : y;\n'
        source += 'end\n'
        return source

    def generate_non_saturating(self):
        source =  f'always @* begin\n'
        source += f'    out_0_0_0 = in_0_0_0 < 0 ? 0 : in_0_0_0;\n'
        source += 'end\n'
        return source

    def generate_neuron(self, id, function):
        source = self.generate_module_header(f'neuron_{id}', 1, 1, 1, 1, 1, 1, 'reg')
        source += function()
        source += 'endmodule\n\n'
        self.add_module(id, source)
        return source

    def generate_activation(self, id, depth, width, height, activationFunctionType):
        source = self.generate_module_header(f'activation_{id}', depth, width, height, depth, width, height)
        neuronId = f'fun{int(activationFunctionType)}'
        neuronFunction = self.generate_sigmoid_approximation
        if activationFunctionType == ActivationFunctionType.NON_SATURATING:
            neuron_function = self.generate_non_saturating
        for i in range(depth):
            for j in range(width):
                for k in range(height):
                    self.generate_neuron(neuronId, neuronFunction)
                    source += self.generate_module_include(f'neuron_{neuronId} neuron_{i}_{j}_{k}',
                                                           1, 1, 1, i, j, k, i, j, k)
        source += '\nendmodule\n\n'
        self.add_module(id, source)
        return source

    def generate_fully_connected_layer(self, id, weights, bias = None):
        inDepth = weights.shape[0]
        inWidth = weights.shape[1]
        inHeight = weights.shape[2]
        outDepth = weights.shape[3]
        outWidth = weights.shape[4]
        outHeight = weights.shape[5]
        source = self.generate_module_header(f'fully_connected_{id}',
                                             inDepth, inWidth, inHeight, outDepth, outWidth, outHeight)
        for i in range(outDepth):
            for j in range(outWidth):
                for k in range(outHeight):
                    convId = f'fc_{id}_{i}_{j}_{k}'
                    convBias = None
                    if bias is not None:
                        convBias = bias[:, i, j, k]
                    self.generate_convolutional_layer(convId, weights[:, :, :, i, j, k], convBias)
                    source += self.generate_module_include(f'convolutional_{convId} connections_{i}_{j}_{k}',
                                                           inDepth, inWidth, inHeight, 0, 0, 0, i, j, k)
        source += '\nendmodule\n\n'
        self.add_module(id, source)
        return source

    def generate_layer_include(self, layer, num, last):
        source = ''
        if layer.type == LayerType.CONVOLUTIONAL:
            self.generate_convolutional_layer(layer.id, layer.weights, layer.bias)
            source = f'convolutional_{layer.id}'
        elif layer.type == LayerType.MAXPOOL:
            self.generate_maxpool_layer(layer.id, layer.inDepth, layer.inWidth, layer.inHeight)
            source = f'maxpool_{layer.id}'
        elif layer.type == LayerType.ACTIVATION:
            self.generate_activation(layer.id, layer.inDepth, layer.inWidth, layer.inHeight, layer.funType)
            source = f'activation_{layer.id}'
        elif layer.type == LayerType.DENSE:
            self.generate_dense_layer(layer.id, layer.inDepth, layer.inWidth, layer.inHeight,
                                      layer.filterWidth, layer.filterHeight, layer.stepX, layer.stepY, layer.filters)
            source = f'dense_{layer.id}'
        elif layer.type == LayerType.FULLY_CONNECTED:
            self.generate_fully_connected_layer(layer.id, layer.weights, layer.bias)
            source = f'fully_connected_{layer.id}'
        source += f' layer_{num}(.clk(clk), .rst(rst)'

        fromFun = lambda x, y, z: f'con_{num - 1}[{x}][{y}][{z}]'
        if num == 0:
            fromFun = lambda x, y, z: f'in_{x}_{y}_{z}'
        source += self.generate_params_pass(layer.inDepth, layer.inWidth, layer.inHeight, 0, 0, 0,
                                            fromFun, lambda x, y, z: f'in_{x}_{y}_{z}')

        toFun = lambda x, y, z: f'con_{num}[{x}][{y}][{z}]'
        if last:
            toFun = lambda x, y, z: f'out_{x}_{y}_{z}'
        source += self.generate_params_pass(layer.outDepth, layer.outWidth, layer.outHeight, 0, 0, 0,
                                            toFun, lambda x, y, z: f'out_{x}_{y}_{z}')

        source += ');\n'
        return source

    def generate_network(self, layers):
        source = self.generate_module_header('network', layers[0].inDepth, layers[0].inWidth, layers[0].inHeight,
                                             layers[-1].outDepth, layers[-1].outWidth, layers[-1].outHeight)

        for i in range(len(layers) - 1):
            source += f'wire [{self.busWidth - 1}:0] con_{i}[{layers[i].outDepth}][{layers[i].outWidth}][{layers[i].outHeight}];\n'

        source += '\n'

        for i in range(len(layers)):
            source += self.generate_layer_include(layers[i], i, i == len(layers) - 1)

        source += '\nendmodule\n\n'
        self.add_module(id, source)
        return source

    def generate_testbench(self, layers, inputs, outputs = None):
        source =  '\n`define assert_close(expected, got, eps) \\\n'
        source += '$display("TEST in %m: got %d, expected %d", got, expected); \\\n'
        source += 'if ((expected > got && expected > got + eps) || (expected < got && expected + eps < got)) begin \\\n'
        source += '    $stop; \\\n'
        source += 'end\n\n'
        source += 'module example_tb;\n'
        source += 'logic clk;\n'
        source += 'logic rst;\n'

        inDepth = layers[0].inDepth
        inWidth = layers[0].inWidth
        inHeight = layers[0].inHeight
        outDepth = layers[-1].outDepth
        outWidth = layers[-1].outWidth
        outHeight = layers[-1].outHeight

        source += self.generate_params(inDepth, inWidth, inHeight, f'reg signed [{self.busWidth - 1}:0] in', ';\n')
        source += self.generate_params(outDepth, outWidth, outHeight, f'wire signed [{self.busWidth - 1}:0] out', ';\n')

        # network
        source += '\nnetwork net(.clk(clk), .rst(rst)'
        in_fun = lambda x, y, z: f'in_{x}_{y}_{z}'
        out_fun = lambda x, y, z: f'out_{x}_{y}_{z}'
        source += self.generate_params_pass(inDepth, inWidth, inHeight, 0, 0, 0, in_fun, in_fun)
        source += self.generate_params_pass(outDepth, outWidth, outHeight, 0, 0, 0, out_fun, out_fun)
        source += ');\n\n'

        # test
        source += '\ntask test;\n'
        source += f'input signed [{self.busWidth - 1}:0] '
        source += self.generate_params(inDepth, inWidth, inHeight, 'test_in', ', ')
        if outputs is not None:
            source += self.generate_params(outDepth, outWidth, outHeight, 'test_out', ', ')
        source += ' test_num;\n'

        source += 'begin\n'
        source += '$display("Test %d started", test_num);\n'
        for i in range(inDepth):
            for j in range(inWidth):
                for k in range(inHeight):
                    source += f'    in_{i}_{j}_{k} <= test_in_{i}_{j}_{k};\n'
        source += '    #1000000000ns\n'

        for i in range(outDepth):
            for j in range(outWidth):
                for k in range(outHeight):
                    if outputs is None:
                        source += f'    $display(out_{i}_{j}_{k});\n'
                    else:
                        source += f'    `assert_close(test_out_{i}_{j}_{k}, out_{i}_{j}_{k}, {int(self.eps * (10 ** self.decimalPrecision))});\n'

        source += 'end\n'
        source += 'endtask\n'

        # run all tests
        source += '\ninitial\n'
        source += 'begin\n'
        source += '    $dumpfile("waves.vcd");\n'
        source += '    $dumpvars;\n'
        for num in range(0, len(inputs)):
            source += '    test('
            for i in range(inDepth):
                for j in range(inWidth):
                    for k in range(inHeight):
                        source += f'{self.get_decimal(inputs[num][i][j][k])}, '
            if outputs is not None:
                for i in range(outDepth):
                    for j in range(outWidth):
                        for k in range(outHeight):
                            source += f'{self.get_decimal(outputs[num][i][j][k])}, '
            source += f'{num});\n'
        source += '    $display("SUCCESS!");\n'
        source += 'end\n'
        source += 'endmodule\n\n'
        self.add_module('testbench', source)
        return source

    def get_source(self):
        source = ''
        for module in self.modules.values():
            source += module
        return source

if __name__ == '__main__':
    generator = FPGANetworkGenerator()

    conv = ConvolutionalLayer(0, np.random.random((1, 2, 2)), np.random.random((1)))
    maxpool = MaxpoolLayer(1, 1, 2, 2, 1, 1)
    denseConv = DenseLayer(2, 1, 4, 4, [conv], 1, 1)
    denseMax = DenseLayer(3, 1, 3, 3, [maxpool], 1, 1)
    activation = ActivationLayer(4, 1, 2, 2, ActivationFunctionType.NON_SATURATING)
    fc = FullyConnectedLayer(5, np.random.random((1, 2, 2, 1, 1, 1)))

    layers = [denseConv, denseMax, activation, fc]

    generator.generate_network(layers)
    generator.generate_testbench(layers, np.random.random((100, 1, 4, 4)))
    print(generator.get_source())
