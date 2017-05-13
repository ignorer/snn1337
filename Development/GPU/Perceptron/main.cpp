#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <chrono>

#define __CL_ENABLE_EXCEPTIONS
#include <cl.hpp>
#include <errCode.h>

#include "FullyConnectedNN.h"
#include "Layer.h"
#include "ClStructHolder.h"

using namespace std;

using Vector = vector<float>;
using IntVector = vector<int>;

FullyConnectedNN loadFullyConnectedNN(const string& filename) {
    ifstream in(filename);

    vector<Layer> layers;

    string strWidth;
    while (getline(in, strWidth)) {
        istringstream inWidth(strWidth);
        int width;
        inWidth >> width;
        string strWeights;
        vector<Vector> weights;
        for (int i = 0; i < width; i++) {
            getline(in, strWeights);
            istringstream iss(strWeights);
            Vector neuronWeights{istream_iterator<float>{iss}, istream_iterator<float>{}};
            weights.push_back(neuronWeights);
        }
        string strBiases;
        getline(in, strBiases);
        istringstream iss(strBiases);
        Vector biases{istream_iterator<float>{iss}, istream_iterator<float>{}};
        layers.push_back(Layer(width, weights, biases));
    }
    return FullyConnectedNN(layers);
}

ClStructHolder buildClHolder(string kernelFileName, const IntVector& layerSizes, const Vector& weights,
        const char* functionName) {
    cl::Device defaultDevice = cl::Device::getDefault();
    cl::Context context(defaultDevice);
    cl::CommandQueue queue(context, defaultDevice);

    ifstream sourceFile(kernelFileName);
    string sourceCode(istreambuf_iterator<char>(sourceFile), (istreambuf_iterator<char>()));
    cl::Program::Sources source(1, make_pair(sourceCode.c_str(), sourceCode.length() + 1));
    cl::Program program = cl::Program(context, source);

    try {
        program.build({defaultDevice});
    }
    catch (cl::Error& error) {
        string buildLog;
        program.getBuildInfo(defaultDevice, CL_PROGRAM_BUILD_LOG, &buildLog);
        throw runtime_error(buildLog);
    }

    cl::Kernel kernel(program, functionName);

    size_t threadNumber = accumulate(layerSizes.begin() + 1, layerSizes.end(), 0ul);

    return ClStructHolder(context, queue, kernel, threadNumber);
}

void processSingleInput(ClStructHolder& holder, IntVector& layerSizes, Vector& weights, Vector& values,
        Vector& input, Vector& output, IntVector& counters) {
    cl::Context context = holder.getContext();
    cl::CommandQueue queue = holder.getQueue();
    cl::Kernel kernel = holder.getKernel();

    cl::Buffer countersBuffer(context, counters.begin(), counters.end(), true);
    cl::Buffer valuesBuffer(context, values.begin(), values.end(), true);
    cl::Buffer inputBuffer(context, input.begin(), input.end(), true);
    cl::Buffer outputBuffer(context, output.begin(), output.end(), false);

    kernel.setArg(2, countersBuffer);
    kernel.setArg(3, valuesBuffer);
    kernel.setArg(4, inputBuffer);
    kernel.setArg(5, outputBuffer);
    kernel.setArg(6, (int) values.size());
    kernel.setArg(7, (int) layerSizes.size());

    // TODO: fix working group size - 1 is extremely inefficient
    queue.enqueueNDRangeKernel(holder.getKernel(), cl::NullRange, holder.getGlobalRange(), cl::NDRange(1));
    queue.finish();

    cl::copy(queue, outputBuffer, output.begin(), output.end());
}

vector<Vector> processMultipleInputs(ClStructHolder& holder, IntVector& layerSizes, Vector& weights,
        vector<Vector>& inputs) {
    if (layerSizes.size() == 0) {
        return {};
    }
    size_t batchSize = inputs.size();
    vector<Vector> outputs(batchSize, Vector((size_t) layerSizes.back()));

    int neuronsNumber = accumulate(layerSizes.begin(), layerSizes.end(), 0);
    Vector values((neuronsNumber + layerSizes.size() - 1) * batchSize, 0); // + size - 1 because of biases

    // pack batch into values buffer and init biases
    for (int i = 0; i < batchSize; ++i) {
        memcpy(values.data() + (layerSizes[0] + 1) * i + 1, inputs[i].data(), sizeof(float) * layerSizes[0]);
    }
    int biasIndex = 0;
    for (int i = 0; i < layerSizes.size() - 1; ++i) {
        for (int j = 0; j < batchSize; ++j) {
            values[biasIndex] = 1;
            biasIndex += layerSizes[i] + 1;
        }
    }

    cl::Context context = holder.getContext();
    cl::CommandQueue queue = holder.getQueue();
    cl::Kernel kernel = holder.getKernel();

    // prepare buffers
    cl::Buffer valuesBuffer(context, values.begin(), values.end(), true);

    // bind arguments
    kernel.setArg(3, valuesBuffer);
    kernel.setArg(4, (int) batchSize);

    // run kernel
    int weightsOffset = 0;
    int valuesOffset = 0;
    for (int layerId = 0; layerId < layerSizes.size() - 1; ++layerId) {
        kernel.setArg(5, layerId);
        kernel.setArg(6, weightsOffset);
        kernel.setArg(7, valuesOffset);

        queue.enqueueNDRangeKernel(holder.getKernel(), cl::NullRange, (size_t) layerSizes[layerId + 1]);
        queue.finish();

        weightsOffset += (layerSizes[layerId] + 1) * layerSizes[layerId + 1];
        valuesOffset += (layerSizes[layerId] + 1) * batchSize;
    }

    cl::copy(queue, valuesBuffer, values.begin(), values.end());

    // unpack results of batch processing from values buffer
    float* outputsPtr = values.data() + values.size() - outputs.size() * layerSizes.back();
    for (int i = 0; i < outputs.size(); ++i) {
        memcpy(outputs[i].data(), outputsPtr + i * layerSizes.back(), sizeof(float) * layerSizes.back());
    }
    return outputs;
}

void testXor() {
    FullyConnectedNN network = loadFullyConnectedNN("network_xor");

    IntVector layerSizes = network.getSizes();
    Vector weights = network.getAllWeights();
    Vector values = network.getEmptyValues();
    vector<Vector> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<Vector> expectedOutputs = {{0}, {1}, {1}, {0}};
    IntVector counters(layerSizes.begin(), layerSizes.end());
    counters[0] = 0;

    try {
        ClStructHolder holder = buildClHolder("neuron.cl", layerSizes, weights, "neuron");
        cl::Buffer layersBuffer(holder.getContext(), layerSizes.begin(), layerSizes.end(), true);
        cl::Buffer weightsBuffer(holder.getContext(), weights.begin(), weights.end(), true);
        holder.getKernel().setArg(0, layersBuffer);
        holder.getKernel().setArg(1, weightsBuffer);

        auto start = chrono::steady_clock::now();
        for (size_t i = 0; i < inputs.size() * 1000; ++i) {
            Vector input;
            input.push_back(1);
            input.insert(input.end(), inputs[i % inputs.size()].begin(), inputs[i % inputs.size()].end());
            Vector predictedOutput(layerSizes.back());
            processSingleInput(holder, layerSizes, weights, values, input, predictedOutput, counters);

            bool isOutputCorrect = true;
            for (int j = 0; j < predictedOutput.size(); ++j) {
                isOutputCorrect = isOutputCorrect && (predictedOutput[j] - expectedOutputs[i % inputs.size()][j]) < 1e-5;
            }
//            cout << i << ": " << (isOutputCorrect ? "correct" : "wrong") << endl;
        }
        cout << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << endl;

    } catch (const cl::Error& e) {
        cerr << errCode(e.err()) << endl;
    } catch (const exception& e) {
        cerr << e.what() << endl;
    }
}

void testDigits() {
    FullyConnectedNN network = loadFullyConnectedNN("network_digits");

    IntVector layerSizes = network.getSizes();
    Vector weights = network.getAllWeights();
    Vector values = network.getEmptyValues();
    vector<Vector> inputs = network.getInput("input_digits");
    vector<Vector> expectedOutputs = network.getOutput("output_digits");
    IntVector counters(layerSizes.begin(), layerSizes.end());
    counters[0] = 0;

    try {
        ClStructHolder holder = buildClHolder("neuron.cl", layerSizes, weights, "neuron");
        cl::Buffer layersBuffer(holder.getContext(), layerSizes.begin(), layerSizes.end(), true);
        cl::Buffer weightsBuffer(holder.getContext(), weights.begin(), weights.end(), true);
        holder.getKernel().setArg(0, layersBuffer);
        holder.getKernel().setArg(1, weightsBuffer);

        auto start = chrono::steady_clock::now();
        int imagesNumber = 10000;
        int correctOutputNumber = 0;

        for (size_t i = 0; i < imagesNumber; ++i) {
            Vector input;
            input.push_back(1);
            input.insert(input.end(), inputs[i % inputs.size()].begin(), inputs[i % inputs.size()].end());
            Vector predictedOutput(layerSizes.back());
            processSingleInput(holder, layerSizes, weights, values, input, predictedOutput, counters);

            bool isOutputCorrect = true;
            for (int j = 0; j < predictedOutput.size(); ++j) {
                isOutputCorrect =
                        isOutputCorrect && (predictedOutput[j] - expectedOutputs[i % inputs.size()][j]) < 0.45;
            }
            if (isOutputCorrect) {
                ++correctOutputNumber;
            }
        }
        cout << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << endl;
        cout << float(correctOutputNumber) / imagesNumber << endl;

    } catch (const cl::Error& e) {
        cerr << errCode(e.err()) << endl;
    } catch (const exception& e) {
        cerr << e.what() << endl;
    }
}

void testDigitsBatched() {
    FullyConnectedNN network = loadFullyConnectedNN("network_digits");

    IntVector layerSizes = network.getSizes();
    Vector weights = network.getAllWeights();
    Vector values = network.getEmptyValues();
    vector<Vector> inputs = network.getInput("input_digits");
    vector<Vector> expectedOutputs = network.getOutput("output_digits");

    try {
        ClStructHolder holder = buildClHolder("batchedNeuron.cl", layerSizes, weights, "batchedNeuron");
        cl::Buffer layersBuffer(holder.getContext(), layerSizes.begin(), layerSizes.end(), true);
        cl::Buffer weightsBuffer(holder.getContext(), weights.begin(), weights.end(), true);
        holder.getKernel().setArg(0, layersBuffer);
        holder.getKernel().setArg(1, (int) layerSizes.size());
        holder.getKernel().setArg(2, weightsBuffer);

        auto start = chrono::steady_clock::now();
        int imagesNumber = 10000;
        int correctOutputNumber = 0;

        for (int batch = 0 ; batch < 90; ++batch) {
            auto outputs = processMultipleInputs(holder, layerSizes, weights, inputs);
            for (size_t i = 0; i < outputs.size(); ++i) {
                Vector predictedOutput = outputs[i % inputs.size()];
                Vector expectedOutput = expectedOutputs[i % inputs.size()];

                bool isOutputCorrect = true;
                for (int j = 0; j < predictedOutput.size(); ++j) {
                    isOutputCorrect = isOutputCorrect && (predictedOutput[j] - expectedOutput[j]) < 0.45;
                }
                if (isOutputCorrect) {
                    ++correctOutputNumber;
                }
            }
        }
        cout << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << endl;
//        cout << float(correctOutputNumber) / outputs.size() << endl;

    } catch (const cl::Error& e) {
        cerr << errCode(e.err()) << endl;
    } catch (const exception& e) {
        cerr << e.what() << endl;
    }
}

int main() {
    testDigitsBatched();
    return 0;
}
