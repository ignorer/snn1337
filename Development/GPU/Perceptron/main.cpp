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
#include "InputReader.h"

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

vector<vector<float>> processMultipleInputs(ClStructHolder& holder, vector<int>& layerSizes, vector<float>& weights,
        vector<vector<float>>& inputs, int batchSize) {
    if (layerSizes.size() == 0) {
        return {};
    }
    vector<vector<float>> outputs(batchSize, vector<float>((size_t) layerSizes.back()));

    int neuronsNumber = accumulate(layerSizes.begin(), layerSizes.end(), 0);
    vector<float> values((neuronsNumber + layerSizes.size() - 1) * batchSize, 0); // + size - 1 because of biases

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
    kernel.setArg(4, batchSize);

    // run kernel
    int weightsOffset = 0;
    int valuesOffset = 0;
    for (int layerId = 0; layerId < layerSizes.size() - 1; ++layerId) {
        kernel.setArg(5, layerId);
        kernel.setArg(6, weightsOffset);
        kernel.setArg(7, valuesOffset);

        // next two lines are effective only for advanced gpu
//        kernel.setArg(8, cl::Local(sizeof(float) * (layerSizes[layerId] + 1)));
//        queue.enqueueNDRangeKernel(holder.getKernel(), cl::NullRange, (size_t) layerSizes[layerId + 1] * 10, 1);
        queue.enqueueNDRangeKernel(holder.getKernel(), cl::NullRange, (size_t) layerSizes[layerId + 1], cl::NullRange);
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

void testDigitsBatched() {
    FullyConnectedNN network = loadFullyConnectedNN("network_mnist");
    InputReader inputReader;
    inputReader.read();

    vector<int> layerSizes = network.getSizes();
    vector<float> weights = network.getAllWeights();
    vector<float> values = network.getEmptyValues();
    vector<vector<float>> inputs = inputReader.getTestImageFloatData();
    vector<int> expectedOutputs = inputReader.getTestImagesLabels();

    try {
        ClStructHolder holder = buildClHolder("batchedNeuron.cl", layerSizes, weights, "batchedNeuron");
        cl::Buffer layersSizesBuffer(holder.getContext(), layerSizes.begin(), layerSizes.end(), true);
        cl::Buffer weightsBuffer(holder.getContext(), weights.begin(), weights.end(), true);
        holder.getKernel().setArg(0, layersSizesBuffer);
        holder.getKernel().setArg(1, (int) layerSizes.size());
        holder.getKernel().setArg(2, weightsBuffer);

        auto start = chrono::steady_clock::now();
        int correctOutputNumber = 0;
        int batchSize = (int) inputs.size();

        auto outputs = processMultipleInputs(holder, layerSizes, weights, inputs, batchSize);

        for (size_t i = 0; i < batchSize; ++i) {
            vector<float>& predictedOutputs = outputs[i];

            int expectedOutput = expectedOutputs[i];

            int predictedOutput = (int) (max_element(predictedOutputs.begin(), predictedOutputs.end())
                    - predictedOutputs.begin());
            if (predictedOutput == expectedOutput) {
                ++correctOutputNumber;
            }
        }
        cout << "images number: " << inputs.size() << endl;
        cout << "elapsed time: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << endl;
        cout << "accuracy: " << float(correctOutputNumber) / inputs.size() << endl;


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