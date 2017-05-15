#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>

#define __CL_ENABLE_EXCEPTIONS
#include <cl.hpp>
#include <errCode.h>
#include <iterator>
#include <algorithm>
#include <numeric>

#include "Layer.h"
#include "ClStructHolder.h"
#include "FullyConnectedNN.h"
#include "NetworkInitializer.h"
#include "InputReader.h"

using namespace std;

ClStructHolder buildCLHolder(const char* kernelFileName, vector<float>& weights, vector<int> sizes,
                             const char* functionName){
    cl::Device defaultDevice = cl::Device::getDefault();
    cl::Context context(defaultDevice);
    cl::CommandQueue queue(context, defaultDevice);

    ifstream sourceFile(kernelFileName);
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

    size_t threadNumber = 0;
    for(int i = 0; i < sizes.size(); ++i){
        threadNumber += sizes[i];
    }

    return ClStructHolder(context, queue, kernel, threadNumber);
}

void processSingleInput(ClStructHolder& holder, vector<int>& sizes, vector<float>& input, vector<float>& output, vector<int> spikes) {
    cl::Context context = holder.getContext();
    cl::CommandQueue queue = holder.getQueue();
    cl::Kernel kernel = holder.getKernel();

    cl::Buffer spikesBuffer(context, spikes.begin(), spikes.end(), true);
    cl::Buffer inputBuffer(context, input.begin(), input.end(), true);
    cl::Buffer outputBuffer(context, output.begin(), output.end(), false);

    kernel.setArg(11, inputBuffer);
    kernel.setArg(12, outputBuffer);
    kernel.setArg(6, spikesBuffer);
    kernel.setArg(1, (int) sizes.size());

    queue.enqueueNDRangeKernel(holder.getKernel(), cl::NullRange, holder.getGlobalRange(), cl::NDRange(1));
    queue.finish();

    cl::copy(queue, outputBuffer, output.begin(), output.end());
}

void testNetwork(std::string testName, float precision) {
    NetworkInitializer ni(testName);
    FullyConnectedNN network = ni.getNetwork();

    InputReader ir;
    ir.read();
    vector<vector<int>> trainImagesData = ir.getTestImagesData();
    vector<vector<int>> inputs ;
    for(int i = 0; i < trainImagesData.size(); ++i) {
        inputs.push_back(ir.getFrequencies(trainImagesData[i]));
    }

    vector<int> layerSizes = network.getSizes();
    vector<float> weights = network.getAllWeights();
    vector<vector<int>> expectedOutputs ;
    for (int i = 0; i < trainImagesData.size(); ++i) {
        expectedOutputs.push_back(ir.getTestImagesLabels());
    }
    vector<int> spikes;
    vector<float> potentials;
    vector<int> t(1);
    vector<int> sem(1);
    float threshold = network.getThreshold();
    int synapsesPerConnection, spikesPerSynapse, exitTime;
    synapsesPerConnection = network.getSynapsesPerConnection();
    spikesPerSynapse = network.getMaxSpikesPerSynapse();
    exitTime = network.getExitTime();
    unsigned int numberOfSpikes = 0;
    for(int i = 0; i < layerSizes.size()-1; ++i){
        numberOfSpikes += layerSizes[i]*layerSizes[i+1];
    }
    numberOfSpikes *= synapsesPerConnection*spikesPerSynapse;
    spikes.resize(numberOfSpikes, -100);
    int maxSpikesInLayer = *max_element(layerSizes.begin(), layerSizes.end()) * synapsesPerConnection * spikesPerSynapse;
    potentials.resize((unsigned int)maxSpikesInLayer, 0.);

    try {
        ClStructHolder holder = buildCLHolder("neuron.cl", weights, layerSizes, "neuron");
        cl::Buffer layersBuffer(holder.getContext(), layerSizes.begin(), layerSizes.end(), true);
        cl::Buffer weightsBuffer(holder.getContext(), weights.begin(), weights.end(), true);
        cl::Buffer tBuffer(holder.getContext(), t.begin(), t.end(), true);
        cl::Buffer semBuffer(holder.getContext(), sem.begin(), sem.end(), true);
        cl::Buffer potentialsBuffer(holder.getContext(), potentials.begin(), potentials.end(), true);
        holder.getKernel().setArg(0, layersBuffer);
        holder.getKernel().setArg(2, synapsesPerConnection);
        holder.getKernel().setArg(3, spikesPerSynapse);
        holder.getKernel().setArg(4, exitTime);
        holder.getKernel().setArg(5, weightsBuffer);
        holder.getKernel().setArg(7, potentialsBuffer);
        holder.getKernel().setArg(8, threshold);
        holder.getKernel().setArg(9, tBuffer);
        holder.getKernel().setArg(10, semBuffer);

        auto start = chrono::steady_clock::now();
        int correctOutputNumber = 0;
        int imagesNumber = inputs.size();
        for (size_t i = 0; i < inputs.size(); ++i) {
            vector<float> input;
            input.push_back(1);
            input.insert(input.end(), inputs[i].begin(), inputs[i].end());
            vector<float> predictedOutput(layerSizes.back());
            processSingleInput(holder, layerSizes, input, predictedOutput, spikes);

            bool isOutputCorrect = true;
            for (int j = 0; j < predictedOutput.size(); ++j) {
                isOutputCorrect = isOutputCorrect && abs(predictedOutput[j] - expectedOutputs[i][j]) < precision;
            }
            if (isOutputCorrect) {
                ++correctOutputNumber;
            }
        }
        cout << "Testing \"" << testName << "\" with precision " << precision << "..." << endl;
        cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << endl;
        cout << "Accuracy: " << float(correctOutputNumber) / imagesNumber * 100.0 << endl;
    } catch (const cl::Error& e) {
        cerr << errCode(e.err()) << endl;
    } catch (const exception& e) {
        cerr << e.what() << endl;
    }
}

int main(){
//    testNetwork("res/network_test", 1e-5);
    testNetwork("res/network", 0.45);
    return 0;
}