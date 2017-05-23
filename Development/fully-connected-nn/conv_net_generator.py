import numpy as np
from enum import Enum, IntEnum
from keras import layers
from keras.models import Sequential

class FPGANetworkGenerator:
    def __init__(self, busWidth=40, decimalPrecision=4, eps=0.1):
        self.busWidth = busWidth
        self.extendedWidth = busWidth * 2
        self.decimalPrecision = decimalPrecision
        self.eps = eps
        self.modules = {}
        self.activationFunctions = {
            'sigmoid' : self.generate_sigmoid_approximation,
            'relu' : self.generate_non_saturating
        }

    def add_module(self, id, source):
        self.modules[id] = source

    def get_decimal(self, num):
        return int(num * (10 ** self.decimalPrecision) + 0.5)

    @staticmethod
    def generate_params(shape, prefix, suffix = ''):
        source = ''
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    source += f'{prefix}_{i}_{j}_{k}{suffix}'
        return source

    @staticmethod
    def generate_params_pass(shape, pos, nameFrom, nameTo):
        source = ''
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    source += f', .{nameTo(i, j, k)}({nameFrom(i + pos[0], j + pos[1], k + pos[2])})'
        return source

    def generate_module_header(self, name, inShape, outShape, reg = ''):
        source = f'module {name}(clk, rst'
        source += self.generate_params(inShape, ', in')
        source += self.generate_params(outShape, ', out')
        source += ');\n'

        source += 'input wire clk;\n'
        source += 'input wire rst;\n\n'

        source += self.generate_params(inShape, f'input signed [{self.busWidth - 1}:0] in', ';\n')
        source += '\n'
        source += self.generate_params(outShape, f'output {reg} signed [{self.busWidth - 1}:0] out', ';\n')
        source += '\n'
        return source

    def generate_module_include(self, name, inShape, inPos, outPos):
        source = f'{name}(.clk(clk), .rst(rst)'
        name = lambda x, y, z: f'in_{x}_{y}_{z}'
        source += self.generate_params_pass(inShape, inPos, name, name)
        source += f', .out_0_0_0(out_{outPos[0]}_{outPos[1]}_{outPos[2]}));\n'
        return source

    def generate_apply2d_layer(self, id, layer, inShape, outShape):
        filterStep = layer.strides
        filterShape = None
        currDepth = None

        if isinstance(layer, layers.Conv2D):
            filterShape = (inShape[0], layer.kernel_size[0], layer.kernel_size[1])
            currDepth = lambda x: 0
        elif isinstance(layer, layers.MaxPool2D):
            filterShape = (1, layer.pool_size[0], layer.pool_size[1])
            currDepth = lambda x: x
        else:
            assert False, 'This layer is not supported'

        source = self.generate_module_header(f'apply2d_{id}', inShape, outShape)

        for i in range(outShape[0]):
            filterId = None
            filterParams = ''
            filterFunctions = []
            poolFunction = None
            if isinstance(layer, layers.Conv2D):
                filterId = f'{id}_{i}'
                for x in range(filterShape[0]):
                    for y in range(filterShape[1]):
                        for z in range(filterShape[2]):
                            currWeight = layer.get_weights()[0][filterShape[1] - 1 - y, filterShape[2] - 1 - z, x, i]
                            filterParams += f'parameter signed W_{x}_{y}_{z} = {self.get_decimal(currWeight)};\n'
                            filterFunctions.append(f'W_{x}_{y}_{z} * in_{x}_{y}_{z} / {self.get_decimal(1.0)}')
                if layer.use_bias:
                    filterParams += f'parameter signed BIAS = {self.get_decimal(layer.get_weights()[1][i])};\n'
                    filterFunctions.append(f'BIAS')
                poolFunction = lambda x, y: f'({x} + {y})'
            elif isinstance(layer, layers.MaxPool2D):
                filterId = f'{id}_0'
                for x in range(filterShape[0]):
                    for y in range(filterShape[1]):
                        for z in range(filterShape[2]):
                            filterFunctions.append(f'in_{x}_{y}_{z}')
                poolFunction = lambda x, y: f'({x} > {y} ? {x} : {y})'
            else:
                assert False, 'This layer is not supported'
            for j in range(0, outShape[1]):
                for k in range(0, outShape[2]):
                    filterName = f'filter_{filterId} filter_{i}_{j}_{k}'
                    self.generate_filter(filterId, filterShape, filterParams, filterFunctions, poolFunction)
                    source += self.generate_module_include(filterName, filterShape,
                                                           (currDepth(i), j * filterStep[0], k * filterStep[1]),
                                                           (i, j, k))

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

    def generate_filter(self, id, shape, params, functions, poolFunction):
        if id in self.modules.keys():
            return self.modules[id]
        source = self.generate_module_header(f'filter_{id}', shape, (1, 1, 1), 'reg')
        source += params
        tmp = self.generate_pool('out_0_0_0', functions, poolFunction)
        source += tmp[1]
        source += 'always @* begin\n'
        source += tmp[0]
        source += 'end\nendmodule\n\n'
        self.add_module(id, source)
        return source

    def generate_convolutional_filter(self, id, weights, bias):
        filterShape = weights.shape
        filterParams = ''
        filterFunctions = []
        for i in range(filterShape[0]):
            for j in range(filterShape[1]):
                for k in range(filterShape[2]):
                    filterParams += f'parameter signed W_{i}_{j}_{k} = {self.get_decimal(weights[i][j][k])};\n'
                    filterFunctions.append(f'W_{i}_{j}_{k} * in_{i}_{j}_{k} / {self.get_decimal(1.0)}')
        if bias is not None:
            filterParams += f'parameter signed BIAS = {self.get_decimal(bias)};\n'
            filterFunctions.append(f'BIAS')
        poolFunction = lambda x, y: f'({x} + {y})'
        return self.generate_filter(id, filterShape, filterParams, filterFunctions, poolFunction)


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
        source += f'end\n'
        return source

    def generate_neuron(self, id, function):
        source = self.generate_module_header(f'neuron_{id}', (1, 1, 1), (1, 1, 1), 'reg')
        source += function()
        source += 'endmodule\n\n'
        self.add_module(id, source)
        return source

    def generate_activation_layer(self, id, layer, shape):
        assert layer.activation.__name__ in self.activationFunctions, 'This function is not supported'
        neuronId = f'fun_{layer.activation.__name__}'
        neuronFunction = self.activationFunctions[layer.activation.__name__]

        source = self.generate_module_header(f'activation_{id}', shape, shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    self.generate_neuron(neuronId, neuronFunction)
                    source += self.generate_module_include(f'neuron_{neuronId} neuron_{i}_{j}_{k}',
                                                           (1, 1, 1), (i, j, k), (i, j, k))
        source += '\nendmodule\n\n'
        self.add_module(id, source)
        return source

    def generate_reshape_layer(self, id, layer, inShape, outShape):
        source = self.generate_module_header(f'reshape_{id}', inShape, outShape)

        for i in range(inShape[0]):
            for j in range(inShape[1]):
                for k in range(inShape[2]):
                    sum = i * inShape[1] * inShape[2] + j * inShape[2] + k
                    (x, y, z) = (sum // outShape[1] // outShape[2], sum // outShape[2] % outShape[1], sum % outShape[2])
                    source += f'assign out_{x}_{y}_{z} = in_{i}_{j}_{k};\n'

        source += '\nendmodule\n\n'
        self.add_module(id, source)
        return source

    def generate_dense_layer(self, id, layer, inShape, outShape):
        assert inShape[0] == 1 and inShape[1] == 1 , 'Layer shold be flat'
        source = self.generate_module_header(f'dense_{id}', inShape, outShape)
        for i in range(outShape[2]):
            convId = f'fc_{id}_{i}'
            convBias = None
            if layer.use_bias:
                convBias = layer.get_weights()[1]
            self.generate_convolutional_filter(convId, layer.get_weights()[0][:, i].reshape(1, 1, inShape[2]), convBias[i])
            source += self.generate_module_include(f'filter_{convId} connections_{i}',
                                                   inShape, (0, 0, 0), (0, 0, i))

        source += '\nendmodule\n\n'
        self.add_module(id, source)
        return source

    def generate_layer_include(self, id, layer, inShape, outShape, num, last):
        source = ''
        if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.MaxPool2D):
            assert layer.data_format == 'channels_first', 'First dimension should be depth'
            self.generate_apply2d_layer(id, layer, inShape, outShape)
            source = f'apply2d_{id}'
        elif isinstance(layer, layers.Activation):
            assert inShape == outShape, 'Input and output shapes should be equal'
            self.generate_activation_layer(id, layer, inShape)
            source = f'activation_{id}'
        elif isinstance(layer, layers.Dense):
            self.generate_dense_layer(id, layer, inShape, outShape)
            source = f'dense_{id}'
        elif isinstance(layer, layers.Reshape):
            self.generate_reshape_layer(id, layer, inShape, outShape)
            source = f'reshape_{id}'
        else:
            assert False, 'This layer type is not supported'

        source += f' layer_{num}(.clk(clk), .rst(rst)'

        fromFun = lambda x, y, z: f'con_{num - 1}[{x}][{y}][{z}]'
        if num == 0:
            fromFun = lambda x, y, z: f'in_{x}_{y}_{z}'
        source += self.generate_params_pass(inShape, (0, 0, 0), fromFun, lambda x, y, z: f'in_{x}_{y}_{z}')

        toFun = lambda x, y, z: f'con_{num}[{x}][{y}][{z}]'
        if last:
            toFun = lambda x, y, z: f'out_{x}_{y}_{z}'
        source += self.generate_params_pass(outShape, (0, 0, 0), toFun, lambda x, y, z: f'out_{x}_{y}_{z}')

        source += ');\n'
        return source

    def generate_network(self, model):
        source = self.generate_module_header('network', model.get_input_shape_at(0)[1:],
                                             model.get_output_shape_at(0)[1:])

        inShape = []
        outShape = []
        currShape = model.get_input_shape_at(0)
        for i in range(len(model.layers)):
            inShape.append(currShape[1:])
            currShape = model.layers[i].compute_output_shape(currShape)
            outShape.append(currShape[1:])


        for i in range(len(model.layers) - 1):
            source += f'wire [{self.busWidth - 1}:0] con_{i}'\
                      f'[{outShape[i][0]}]'\
                      f'[{outShape[i][1]}]'\
                      f'[{outShape[i][2]}];\n'

        source += '\n'
        num = 0
        for i in range(len(model.layers)):
            if isinstance(model.layers[i], layers.Dropout):
                continue
            source += self.generate_layer_include(num, model.layers[i], inShape[i], outShape[i], num, i == len(model.layers) - 1)
            num += 1

        source += '\nendmodule\n\n'
        self.add_module(id, source)
        return source

    def generate_testbench(self, model, inputs, calcOutputs = False):
        source =  '\n`define assert_close(expected, got, eps) \\\n'
        source += '$display("TEST in %m: got %d, expected %d", got, expected); \\\n'
        source += 'if ((expected > got && expected > got + eps) || (expected < got && expected + eps < got)) begin \\\n'
        source += '    $stop; \\\n'
        source += 'end\n\n'
        source += 'module example_tb;\n'
        source += 'logic clk;\n'
        source += 'logic rst;\n'

        inShape = model.get_input_shape_at(0)[1:]
        outShape = model.get_output_shape_at(0)[1:]
        assert inShape == inputs.shape[1:], 'Incorrect input shape'

        source += self.generate_params(inShape, f'reg signed [{self.busWidth - 1}:0] in', ';\n')
        source += self.generate_params(outShape, f'wire signed [{self.busWidth - 1}:0] out', ';\n')

        # network
        source += '\nnetwork net(.clk(clk), .rst(rst)'
        in_fun = lambda x, y, z: f'in_{x}_{y}_{z}'
        out_fun = lambda x, y, z: f'out_{x}_{y}_{z}'
        source += self.generate_params_pass(inShape, (0, 0, 0), in_fun, in_fun)
        source += self.generate_params_pass(outShape, (0, 0, 0), out_fun, out_fun)
        source += ');\n\n'

        # test
        source += '\ntask test;\n'
        source += f'input signed [{self.busWidth - 1}:0] '
        source += self.generate_params(inShape, 'test_in', ', ')
        if calcOutputs:
            source += self.generate_params(outShape, 'test_out', ', ')
        source += ' test_num;\n'

        source += 'begin\n'
        source += '$display("Test %d started", test_num);\n'
        for i in range(inShape[0]):
            for j in range(inShape[1]):
                for k in range(inShape[2]):
                    source += f'    in_{i}_{j}_{k} <= test_in_{i}_{j}_{k};\n'
        source += '    #1000000000ns\n'

        for i in range(outShape[0]):
            for j in range(outShape[1]):
                for k in range(outShape[2]):
                    if calcOutputs:
                        source += f'    `assert_close(test_out_{i}_{j}_{k}, out_{i}_{j}_{k}, {int(self.eps * (10 ** self.decimalPrecision))});\n'
                    else:
                        source += f'    $display(out_{i}_{j}_{k});\n'

        source += 'end\n'
        source += 'endtask\n'

        # run all tests
        source += '\ninitial\n'
        source += 'begin\n'
        source += '    $dumpfile("waves.vcd");\n'
        source += '    $dumpvars;\n'
        for num in range(0, len(inputs)):
            source += '    test('
            for i in range(inShape[0]):
                for j in range(inShape[1]):
                    for k in range(inShape[2]):
                        source += f'{self.get_decimal(inputs[num][i][j][k])}, '
            if calcOutputs:
                outputs = model.predict(inputs)
                for i in range(outShape[0]):
                    for j in range(outShape[1]):
                        for k in range(outShape[2]):
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
    from keras.datasets import mnist
    from keras.utils import np_utils

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = np_utils.to_categorical(y_train, 10).reshape((y_train.shape[0], 1, 1, 10))
    Y_test = np_utils.to_categorical(y_test, 10).reshape((y_test.shape[0], 1, 1, 10))


    np.random.seed(1337)
    model = Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), input_shape=(1, 28, 28), data_format='channels_first'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), data_format='channels_first'))
    model.add(layers.Conv2D(16, kernel_size=(2, 2), data_format='channels_first'))
    model.add(layers.Activation('relu'))
    model.add(layers.Reshape((1, 1, 2304)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(10))
    model.add(layers.Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
    #model.save('model1337.txt')
    model.load_weights('model1337.txt')
    print(model.evaluate(X_test, Y_test, verbose=0))

    generator = FPGANetworkGenerator()
    generator.generate_network(model)
    generator.generate_testbench(model, X_test[:10], True)
    with open("pinus.sv", "wt") as source:
        source.write(generator.get_source())
