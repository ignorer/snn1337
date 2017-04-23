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


#include <cl.hpp>
#include "CLStructHolder.h"

using namespace std;

CLStructHolder buildCLHolder(string KernelFileName, vector<float>& weights, vector<int> sizes,
                               const char* functionName){
    cl::Device defaultDevice = cl::Device::getDefault();
    cl::Context context(defaultDevice);
    cl::CommandQueue queue(context, defaultDevice);

    ifstream sourceFile(kernelFileName);
    string sourceCode(ifstream_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, make_pair(sourceCode.c_str(), sourceCode.length()+1));
    cl::Program program = cl::Program(context, source);
    try {
        program.built(defaultDevice);
    }
    catch(cl::Error& error) {
        string buildLog;
        program.getBuildInfo(defaultDevice, CL_PROGRAM_BUILD_LOG, &buildLog);
        throw runtime_error(buildLog);
    }

    cl::Kernel kernel(program. functionName);

    size_t threadNumber = 0;
    for(int i = 0; i < sizes.size(); ++i){
        threadNumber += sizes[i];
    }

    return ClStructHolder(context, queue, kernel, threadNumber);
}

void processSignleInput(ClStructHolder& holder, vector<int>& sizes, vector<float>& values, vector<float>& input, vector<float>& output, vector<int>& counters, vector<int> spikes) {
    cl::Context context = holder.getContext();
    cl::CommandQueue queue = holder.getQueue();
    cl::Kernel kernel = holder.getKernel();

    cl::Buffer spikesBuffer(context, spikes.begin(), spikes.end(), true);
    cl::Buffer countersBuffer(context, counters.begin(), counters.end(), true);
    cl::Buffer valuesBuffer(context, values.begin(), values.end(), true);
    cl::Buffer inputBuffer(context, input.begin(), input.end(), true);
    cl::Buffer outputBuffer(context, output.begin(), output.end(), false);

    kernel.setArg(6, countersBuffer);
    kernel.setArg(7, valuesBuffer);
    kernel.setArg(8, inputBuffer);
    kernel.setArg(9, outputBuffer);
    kernel.setArg(10, spikesBuffer);
    kernel.setArg(11, (int) values.size());
    kernel.setArg(12, (int) sizes.size());

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
    vector<int> spikes;
    counters[0] = 0;
    int* t;
    int* sem;
    int synapsesPerConnection, spikesPerSynapse, exitTime;
    int numberOfSpikes = 0;
    for(int i = 0; i < layerSizes.size()-1; ++i){
        numberOfSpikes += layerSizes[i]*layerSizes[i+1];
    }
    numberOfSpikes *= synapsesPerConnection*spikesPerSynapse;
    spikes.resize(numberOfSpikes, -100);
    *t = 0;
    *sem = accumulate(layerSizes.begin(), layerSizes.end(), 0);

    try {
        ClStructHolder holder = buildClHolder("neuron.cl", layerSizes, weights, "neuron");
        cl::Buffer layersBuffer(holder.getContext(), layerSizes.begin(), layerSizes.end(), true);
        cl::Buffer weightsBuffer(holder.getContext(), weights.begin(), weights.end(), true);
        holder.getKernel().setArg(0, layersBuffer);
        holder.getKernel().setArg(1, weightsBuffer);
        holder.getKernel().setArg(2, synapsesPerConnection);
        holder.getKernel().setArg(3, spikesPerSynapse);
        holder.getKernel().setArg(4, *sem);
        holder.getKernel().setArg(5, *t);

        auto start = chrono::steady_clock::now();
        for (size_t i = 0; i < inputs.size(); ++i) {
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
    vector<int> spikes;
    counters[0] = 0;
    int* t;
    int* sem;
    int synapsesPerConnection, spikesPerSynapse, exitTime;
    int numberOfSpikes = 0;
    for(int i = 0; i < layerSizes.size()-1; ++i){
        numberOfSpikes += layerSizes[i]*layerSizes[i+1];
    }
    numberOfSpikes *= synapsesPerConnection*spikesPerSynapse;
    spikes.resize(numberOfSpikes, -100);
    *sem = accumulate(layerSizes.begin(), layerSizes.end(), 0);
    *t = 0;

    try {
        ClStructHolder holder = buildClHolder("neuron.cl", layerSizes, weights, "neuron");
        cl::Buffer layersBuffer(holder.getContext(), layerSizes.begin(), layerSizes.end(), true);
        cl::Buffer weightsBuffer(holder.getContext(), weights.begin(), weights.end(), true);
        holder.getKernel().setArg(0, layersBuffer);
        holder.getKernel().setArg(1, weightsBuffer);
        holder.getKernel().setArg(2, synapsesPerConnection);
        holder.getKernel().setArg(3, spikesPerSynapse);
        holder.getKernel().setArg(4, sem);
        holder.getKernel().setArg(5, t);

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


int main(){

}