#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <numeric>

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
    cl::Program::Sources source(1, make_pair(sourceCode.c_str(), sourceCode.length() + 1));;
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

    cl::Buffer layersBuffer(context, layers.begin(), layers.end(), true);
    cl::Buffer weightsBuffer(context, weights.begin(), weights.end(), true);
    cl::Buffer valuesBuffer(context, values.begin(), values.end(), false);
    cl::Buffer inputBuffer(context, input.begin(), input.end(), true);
    cl::Buffer outputBuffer(context, output.begin(), output.end(), false);
    cl::Buffer countersBuffer(context, counters.begin(), counters.end(), false);

    kernel.setArg(0, layersBuffer);
    kernel.setArg(1, weightsBuffer);
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
    vector<vector<float>> expectedOutputs = {{-1.71589994}, {1.71589994}, {1.71589994}, {-1.71589994}};
    vector<int> counters(layerSizes.begin(), layerSizes.end());
    counters[0] = 0;

    try {
        ClStructHolder clStructValues = buildClHolder("neuron.cl", layerSizes, weights, "neuron");
        for (size_t i = 0; i < inputs.size(); ++i) {
            vector<float> input;
            input.push_back(1);
            input.insert(input.end(), inputs[i].begin(), inputs[i].end());
            vector<float> predictedOutput(layerSizes.back());
            processSignleInput(clStructValues, layerSizes, weights, values, input, predictedOutput, counters);
            cout << i << ": " << (predictedOutput == expectedOutputs[i] ? "correct" : "wrong") << endl;
        }
    } catch (const cl::Error& e) {
        cerr << errCode(e.err()) << endl;
    } catch (const exception& e) {
        cerr << e.what() << endl;
    }
}

int main() {
    testXor();
    return 0;
}