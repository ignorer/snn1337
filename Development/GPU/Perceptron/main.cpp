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

float calculateAccuracyDigits(vector<vector<float>> outputs, vector<int> expectedOutputs,
        int batchSize, int batch_number) {
    float correctOutputNumber = 0;
    for (size_t i = 0; i < batchSize; ++i) {
        vector<float> predictedOutputs = outputs[i];

        int expectedOutput = expectedOutputs[i + batch_number * batchSize];

        int predictedOutput = (int) (max_element(predictedOutputs.begin(), predictedOutputs.end())
                                     - predictedOutputs.begin());
        if (predictedOutput == expectedOutput) {
            ++correctOutputNumber;
        }
    }
    return correctOutputNumber / batchSize;
}

int getMaximumWorkersNumber(int unitsToCompute) {
    return min(CL_DEVICE_MAX_COMPUTE_UNITS, unitsToCompute);
}

int getMaximumWorkGroupSize(int unitsNumber) {
    for (int workGroupSize = CL_DEVICE_MAX_WORK_GROUP_SIZE; workGroupSize > 0; workGroupSize--) {
        if (unitsNumber % workGroupSize == 0) {
            return workGroupSize;
        }
    }
    return -1;
}

vector<vector<float>> processMultipleInputs(ClStructHolder& holder, vector<int>& layerSizes, vector<float>& weights,
        vector<vector<float>>& inputs, int batchSize, int batchNumber, bool workGroupSizeIsOne = false) {
    if (layerSizes.size() == 0) {
        return {};
    }
    vector<vector<float>> outputs(batchSize, vector<float>((size_t) layerSizes.back()));

    int neuronsNumber = accumulate(layerSizes.begin(), layerSizes.end(), 0);
    vector<float> values((neuronsNumber + layerSizes.size() - 1) * batchSize, 0); // + size - 1 because of biases

    // pack batch into values buffer and init biases
    for (int i = 0; i < batchSize; ++i) {
        memcpy(values.data() + (layerSizes[0] + 1) * i + 1, inputs[i + batchNumber * batchSize].data(),
            sizeof(float) * layerSizes[0]);
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

        // TODO: switch to layerSizes[layerId + 1] * batchSize after properly implementing getMaximumWorkersNumber
        int unitsNumber = getMaximumWorkersNumber(layerSizes[layerId + 1]);
        int workGroupSize = 1;
        if (!workGroupSizeIsOne) {
            workGroupSize = getMaximumWorkGroupSize(unitsNumber);
        }

        cout << "unitsNumber for layer " << layerId << ": " << unitsNumber << endl;
        cout << "workGroupSize for layer " << layerId << ": " << workGroupSize << endl;

        if (workGroupSizeIsOne) {
            kernel.setArg(8, cl::Local(sizeof(float) * (layerSizes[layerId] + 1)));
        }
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(unitsNumber), cl::NDRange(workGroupSize));
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

void testDigitsBatched(int batchSize, cl_device_type deviceType, bool workGroupSizeisOne = false) {
    FullyConnectedNN network = loadFullyConnectedNN("network_mnist");
    InputReader ir;
    ir.read();

    vector<int> layerSizes = network.getSizes();
    vector<float> weights = network.getAllWeights();
    vector<float> values = network.getEmptyValues();
    vector<vector<float>> inputs = ir.getTestImageFloatData();
    vector<int> expectedOutputs = ir.getTestImagesLabels();

    try {
        ClStructHolder holder = workGroupSizeisOne ?
                ClStructHolder("batchedNeuron.cl", layerSizes, weights, "batchedNeuronLocal", deviceType) :
                ClStructHolder("batchedNeuron.cl", layerSizes, weights, "batchedNeuron", deviceType);
        cl::Buffer layersSizesBuffer(holder.getContext(), layerSizes.begin(), layerSizes.end(), true);
        cl::Buffer weightsBuffer(holder.getContext(), weights.begin(), weights.end(), true);
        holder.getKernel().setArg(0, layersSizesBuffer);
        holder.getKernel().setArg(1, (int) layerSizes.size());
        holder.getKernel().setArg(2, weightsBuffer);

        int imagesNumber = inputs.size();

        auto start = chrono::steady_clock::now();

        float accuracy = 0;

        for (int i = 0; i < imagesNumber / batchSize; i++) {
            auto outputs = processMultipleInputs(holder, layerSizes, weights, inputs, batchSize, i,
                workGroupSizeisOne);
            accuracy += calculateAccuracyDigits(outputs, expectedOutputs, batchSize, i);
            cout << "batch " << i << " accuracy: " <<  accuracy / (i + 1) << endl;
        }

        cout << "time:" <<
            chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << endl;

    } catch (const cl::Error& e) {
        cerr << errCode(e.err()) << endl;
    } catch (const exception& e) {
        cerr << e.what() << endl;
    }
}

void test(){
    vector<int> results;
    results.resize(2, 0);
    const char filename[] = "batchedNeuronTests.cl";
    const char functionName[] = "tests";
    vector<int> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector<int> b = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    cout << "hello" << endl;

    try{
        cl::Device defaultDevice = cl::Device::getDefault();
        cl::Context context(defaultDevice);
        cl::CommandQueue queue(context, defaultDevice);

        ifstream sourceFile(filename);
        string sourceCode(istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, make_pair(sourceCode.c_str(), sourceCode.length()+1));
        cl::Program program = cl::Program(context, source);
        try {
            program.build({defaultDevice});
        }
        catch(cl::Error& error) {
            string buildLog;
            program.getBuildInfo(defaultDevice, CL_PROGRAM_BUILD_LOG, &buildLog);
            throw runtime_error(buildLog);
        }

        cl::Kernel kernel(program, functionName);

        size_t threadNumber = 1;
        cl::Buffer aBuffer(context, a.begin(), a.end(), true);
        cl::Buffer bBuffer(context, b.begin(), b.end(), true);
        cl::Buffer resultsBuffer(context, results.begin(), results.end(), true);

        kernel.setArg(0, aBuffer);
        kernel.setArg(1, bBuffer);
        kernel.setArg(2, resultsBuffer);

        cout << "hello" << endl;

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, 1, cl::NDRange(1));
        queue.finish();
        cl::copy(queue, resultsBuffer, results.begin(), results.end());
        for(int i = 0; i < 2; ++i){
            cout << results[i] << endl;
            if(results[i] == 0){
                cout << "we have some trouble in function number" << i << endl;
            }
        }
    }catch (const cl::Error& e){
        cerr << errCode(e.err()) << endl;
    }
}

int main() {
    testDigitsBatched(10000, CL_DEVICE_TYPE_GPU, true);
    return 0;
}
