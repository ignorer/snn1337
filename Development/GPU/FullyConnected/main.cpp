#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>

#include <cl.hpp>
#include <numeric>

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

ClStructHolder buildClHolder(string kernelFileName, vector<size_t>& layerSizes, vector<float>& weights,
        const char* functionName, vector<int>& counters) {
    vector<cl::Platform> allPlatforms;
    cl::Platform::get(&allPlatforms);
    if (allPlatforms.size() == 0) {
        cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

    cl::Platform defaultPlatform = allPlatforms[0];
    vector<cl::Device> allDevices;
    defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
    if (allDevices.size() == 0) {
        cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device defaultDevice = allDevices[0];
    cl::Context context(defaultDevice);

    ifstream sourceFile(kernelFileName);
    string sourceCode(istreambuf_iterator<char>(sourceFile), (istreambuf_iterator<char>()));
    cl::Program::Sources source(1, make_pair(sourceCode.c_str(), sourceCode.length() + 1));;
    cl::Program program = cl::Program(context, source);
    if (program.build({defaultDevice}) != CL_SUCCESS) {
        cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice) << "\n";
        getchar();
        exit(1);
    }

    cl::Buffer layerSizesBuffer(context, CL_MEM_READ_WRITE, layerSizes.size() * sizeof(int));
    cl::Buffer weightsBuffer(context, CL_MEM_READ_WRITE, weights.size() * sizeof(float));
    cl::Buffer countersBuffer(context, CL_MEM_READ_WRITE, counters.size() * sizeof(int));

    cl::CommandQueue queue(context, defaultDevice);
    queue.enqueueWriteBuffer(layerSizesBuffer, CL_TRUE, 0, layerSizes.size() * sizeof(int), layerSizes.data());
    queue.enqueueWriteBuffer(weightsBuffer, CL_TRUE, 0, weights.size() * sizeof(float), weights.data());
    queue.enqueueWriteBuffer(countersBuffer, CL_TRUE, 0, counters.size() * sizeof(int), counters.data());

    cl::Kernel kernel(program, functionName);

    kernel.setArg(1, layerSizesBuffer);
    kernel.setArg(2, weightsBuffer);
    kernel.setArg(6, countersBuffer);

    size_t threadNumber = accumulate(layerSizes.begin(), layerSizes.end(), 0ul);
    size_t groupSize = *max_element(layerSizes.begin(), layerSizes.end());

    return ClStructHolder(context, queue, kernel, threadNumber, groupSize);
}

void loopFunction(ClStructHolder holder, vector<float>& values, vector<float>& input, vector<float>& output) {
    cl::Context context = holder.getContext();
    cl::CommandQueue queue = holder.getQueue();
    cl::Kernel kernel = holder.getKernel();

    cl::Buffer valuesBuffer(context, CL_MEM_READ_WRITE, values.size() * sizeof(float));
    cl::Buffer inputBuffer(context, CL_MEM_READ_WRITE, input.size() * sizeof(float));
    cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, output.size() * sizeof(float));

    queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, input.size() * sizeof(float),
            input.data());

    kernel.setArg(3, valuesBuffer);
    kernel.setArg(4, inputBuffer);
    kernel.setArg(5, outputBuffer);

    queue.enqueueNDRangeKernel(holder.getKernel(), cl::NullRange, holder.getGlobalRange(), holder.getLocalRange());
    queue.finish();

    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, output.size() * sizeof(float), output.data());
}

int main() {
    FullyConnectedNN network = loadFullyConnectedNN("network_xor");

    vector<size_t> layerSizes = network.getSizes();
    vector<float> weights = network.getAllWeights();
    vector<float> values = network.getEmptyValues();
    vector<vector<float>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<float>> expectedOutputs = {{0}, {1}, {1}, {0}};
    vector<int> counters(layerSizes.begin() + 1, layerSizes.end());

    ClStructHolder clStructValues = buildClHolder("neuron.cl", layerSizes, weights, "neuron", counters);
    for (size_t i = 0; i < inputs.size(); ++i) {
        vector<float> predictedOutput(layerSizes.back());
        loopFunction(clStructValues, values, inputs[i], predictedOutput);
        cout << i << ": " << (predictedOutput == expectedOutputs[i] ? "correct" : "wrong") << endl;
    }
    return 0;
}