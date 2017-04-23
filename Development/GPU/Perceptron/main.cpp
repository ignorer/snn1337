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

FullyConnectedNN loadFullyConnectedNN(const string& filename) {
    ifstream in(filename);

    vector<Layer> layers;

    string strWidth;
    while (getline(in, strWidth)) {
        istringstream inWidth(strWidth);
        int width;
        inWidth >> width;
        string strWeights;
        vector<vector<float>> weights;
        for (int i = 0; i < width; i++) {
            getline(in, strWeights);
            istringstream iss(strWeights);
            vector<float> neuronWeights{istream_iterator<float>{iss}, istream_iterator<float>{}};
            weights.push_back(neuronWeights);
        }
        string strBiases;
        getline(in, strBiases);
        istringstream iss(strBiases);
        vector<float> biases{istream_iterator<float>{iss}, istream_iterator<float>{}};
        layers.push_back(Layer(width, weights, biases));
    }
    return FullyConnectedNN(layers);
}

ClStructHolder buildClHolder(string kernelFileName, const vector<int>& layerSizes, const vector<float>& weights,
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

void processSignleInput(ClStructHolder& holder, vector<int>& layers, vector<float> weights, vector<float>& values,
        vector<float>& input, vector<float>& output, vector<int>& counters) {
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
    kernel.setArg(7, (int) layers.size());

    // TODO: fix working group size - 1 is extremely inefficient
    queue.enqueueNDRangeKernel(holder.getKernel(), cl::NullRange, holder.getGlobalRange(), cl::NDRange(1));
    queue.finish();

    cl::copy(queue, outputBuffer, output.begin(), output.end());
}

void testXor() {
    FullyConnectedNN network = loadFullyConnectedNN("network_xor");

    vector<int> layerSizes = network.getSizes();
    vector<float> weights = network.getAllWeights();
    vector<float> values = network.getEmptyValues();
    vector<vector<float>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<float>> expectedOutputs = {{0}, {1}, {1}, {0}};
    vector<int> counters(layerSizes.begin(), layerSizes.end());
    counters[0] = 0;

    try {
        ClStructHolder holder = buildClHolder("neuron.cl", layerSizes, weights, "neuron");
        cl::Buffer layersBuffer(holder.getContext(), layerSizes.begin(), layerSizes.end(), true);
        cl::Buffer weightsBuffer(holder.getContext(), weights.begin(), weights.end(), true);
        holder.getKernel().setArg(0, layersBuffer);
        holder.getKernel().setArg(1, weightsBuffer);

        auto start = chrono::steady_clock::now();
        for (size_t i = 0; i < inputs.size() * 1000; ++i) {
            vector<float> input;
            input.push_back(1);
            input.insert(input.end(), inputs[i % inputs.size()].begin(), inputs[i % inputs.size()].end());
            vector<float> predictedOutput(layerSizes.back());
            processSignleInput(holder, layerSizes, weights, values, input, predictedOutput, counters);

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

    vector<int> layerSizes = network.getSizes();
    vector<float> weights = network.getAllWeights();
    vector<float> values = network.getEmptyValues();
    vector<vector<float>> inputs = network.getInput("input_digits");
    vector<vector<float>> expectedOutputs = network.getOutput("output_digits");
    vector<int> counters(layerSizes.begin(), layerSizes.end());
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
            vector<float> input;
            input.push_back(1);
            input.insert(input.end(), inputs[i % inputs.size()].begin(), inputs[i % inputs.size()].end());
            vector<float> predictedOutput(layerSizes.back());
            processSignleInput(holder, layerSizes, weights, values, input, predictedOutput, counters);

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

int main() {
    testDigits();
    return 0;
}
