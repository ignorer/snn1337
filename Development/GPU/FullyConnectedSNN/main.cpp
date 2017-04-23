//
// Created by mariia on 16.04.2017.
//

//1) int* sizes - массив с размерами слоёв. Включает в себя размер входной и выходной слой
//2) int layersNumber - размер массива sizes, просто нужно передать в kernel
//3) int synapsesPerConnection - количество синаптических связей на соединение
//4) int spikesPerSynapse - количество спайков, которые одновременно может помнить один синапс
//5) int exitTime - количество тактов работы сети, после которого та прекращает свою работу
//6) float* weights - массив с весами связей
//7) int* spikes - массив, который содержит время отправления каждого спайка. Изначально времена всех спайков установлены на -inf (например, -100),
// без например, а просто -100,
//8) int* t - указатель на время, изначально равен 0
//9) int* sem - семафор для синхронизации всех нейронов. Изначально равен количеству нейронов вместе со входным и выходным слоями
//10) int* input - буфер, содержащий частоты, с которыми работают нейроны входного слоя
//11) int* output - буфер с временами спайков, выходящих с последнего слоя сети. Его размер равен произведению размера последнего слоя и exitTime


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

#include "Layer.h"
#include "ClStructHolder.h"
#include "FullyConnectedNN.h"
#include "NetworkInitializer.h"
#include "InputReader.h"


typedef int size_type;
// we want typedef unsigned int size_type;
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

void processSignleInput(ClStructHolder& holder, vector<int>& sizes, vector<float>& input, vector<float>& output, vector<int>& counters, vector<int> spikes) {
    cl::Context context = holder.getContext();
    cl::CommandQueue queue = holder.getQueue();
    cl::Kernel kernel = holder.getKernel();

    cl::Buffer spikesBuffer(context, spikes.begin(), spikes.end(), true);
    cl::Buffer countersBuffer(context, counters.begin(), counters.end(), true);
    cl::Buffer inputBuffer(context, input.begin(), input.end(), true);
    cl::Buffer outputBuffer(context, output.begin(), output.end(), false);

//    kernel.setArg(8, countersBuffer);
    kernel.setArg(10, inputBuffer);
    kernel.setArg(11, outputBuffer);
    kernel.setArg(6, spikesBuffer);
    kernel.setArg(1, (int) sizes.size());

    queue.enqueueNDRangeKernel(holder.getKernel(), cl::NullRange, holder.getGlobalRange(), cl::NDRange(1));
    queue.finish();

    cl::copy(queue, outputBuffer, output.begin(), output.end());
}

void testNetworkTest() {
    NetworkInitializer ni("res/network_test");
    FullyConnectedNN network = ni.getNetwork();

    InputReader ir;
    ir.read();
    vector<vector<int>> trainImagesData = ir.getTestImagesData();
    vector<vector<int>> inputs ;
    for(int i = 0; i < trainImagesData.size(); ++i) {
         inputs.push_back(ir.getFrequencies(trainImagesData[i]));
    }

    vector<size_type> layerSizes = network.getSizes();
    vector<float> weights = network.getAllWeights();
    vector<vector<int>> expectedOutputs ;
    for (int i = 0; i < trainImagesData.size(); ++i) {
        expectedOutputs.push_back(ir.getTestImagesLabels());
    }
    vector<int> counters(layerSizes.begin(), layerSizes.end());
    vector<int> spikes;
    counters[0] = 0;
    int* t = new int;
    int* sem = new int;
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
    *t = 0;
    *sem = accumulate(layerSizes.begin(), layerSizes.end(), 0);

    try {
        ClStructHolder holder = buildCLHolder("neuron.cl", weights, layerSizes, "neuron");
        cl::Buffer layersBuffer(holder.getContext(), layerSizes.begin(), layerSizes.end(), true);
        cl::Buffer weightsBuffer(holder.getContext(), weights.begin(), weights.end(), true);
        holder.getKernel().setArg(0, layersBuffer);
        holder.getKernel().setArg(2, synapsesPerConnection);
        holder.getKernel().setArg(3, spikesPerSynapse);
        holder.getKernel().setArg(4, exitTime);
        holder.getKernel().setArg(5, weightsBuffer);
        holder.getKernel().setArg(7, threshold);
        holder.getKernel().setArg(8, t);
        holder.getKernel().setArg(9, sem);

        auto start = chrono::steady_clock::now();
        for (size_t i = 0; i < inputs.size(); ++i) {
            vector<float> input;
            input.push_back(1);
            input.insert(input.end(), inputs[i].begin(), inputs[i].end());
            vector<float> predictedOutput(layerSizes.back());
            processSignleInput(holder, layerSizes, input, predictedOutput, counters, spikes);

            bool isOutputCorrect = true;
            for (int j = 0; j < predictedOutput.size(); ++j) {
                isOutputCorrect = isOutputCorrect && (predictedOutput[j] - expectedOutputs[i][j]) < 1e-5;
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

void testNetwork() {
    NetworkInitializer ni("res/network");
    FullyConnectedNN network = ni.getNetwork();

    InputReader ir;
    ir.read();
    vector<vector<int>> trainImagesData = ir.getTestImagesData();
    vector<vector<int>> inputs ;
    for(int i = 0; i < trainImagesData.size(); ++i) {
        inputs.push_back(ir.getFrequencies(trainImagesData[i]));
    }

    vector<size_type> layerSizes = network.getSizes();
    vector<float> weights = network.getAllWeights();
    vector<vector<int>> expectedOutputs ;
    for (int i = 0; i < trainImagesData.size(); ++i) {
        expectedOutputs.push_back(ir.getTestImagesLabels());
    }
    vector<int> counters(layerSizes.begin(), layerSizes.end());
    vector<int> spikes;
    counters[0] = 0;
    int* t = new int;
    int* sem = new int;
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
    *sem = accumulate(layerSizes.begin(), layerSizes.end(), 0);
    *t = 0;

    try {

        ClStructHolder holder = buildCLHolder("neuron.cl", weights, layerSizes, "neuron");
        cl::Buffer layersBuffer(holder.getContext(), layerSizes.begin(), layerSizes.end(), true);
        cl::Buffer weightsBuffer(holder.getContext(), weights.begin(), weights.end(), true);
        holder.getKernel().setArg(0, layersBuffer);
        holder.getKernel().setArg(2, synapsesPerConnection);
        holder.getKernel().setArg(3, spikesPerSynapse);
        holder.getKernel().setArg(4, exitTime);
        holder.getKernel().setArg(5, weightsBuffer);
        holder.getKernel().setArg(7, threshold);
        holder.getKernel().setArg(8, t);
        holder.getKernel().setArg(9, sem);


        auto start = chrono::steady_clock::now();
        int imagesNumber = 10000;
        int correctOutputNumber = 0;

        for (size_t i = 0; i < imagesNumber; ++i) {
            vector<float> input;
            input.push_back(1);
            input.insert(input.end(), inputs[i % inputs.size()].begin(), inputs[i % inputs.size()].end());
            vector<float> predictedOutput(layerSizes.back());
            processSignleInput(holder, layerSizes, input, predictedOutput, counters, spikes);

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


int main(){
    testNetworkTest();
    testNetwork();
    return 0;
}