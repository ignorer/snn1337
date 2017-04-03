#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>

#include <cl.hpp>

#include "FullyConnectedNN.h"
#include "Layer.h"

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
        vector<vector<double>> weights;
        for (int i = 0; i < width; i++) {
            getline(in, strWeights);
            istringstream iss(strWeights);
            vector<double> neuronWeights{istream_iterator<double>{iss}, istream_iterator<double>{}};
            weights.push_back(neuronWeights);
        }
        string strBiases;
        getline(in, strBiases);
        istringstream iss(strBiases);
        vector<double> biases{istream_iterator<double>{iss}, istream_iterator<double>{}};
        layers.push_back(Layer(width, weights, biases));
    }
    return FullyConnectedNN(layers);
}

    // the example of usage of Olga's classes for NN initialization
//int main() {
//    FullyConnectedNN network = loadFullyConnectedNN("network_xor");
//    network.printEmptyValues();
//}

class clStruct{
private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel kernel;
    cl::NDRange global;
public:
    clStruct(cl::Context context_, cl::CommandQueue queue_, cl::Kernel kernel_, cl::NDRange global_){
        context = context_;
        queue = queue_;
        kernel = kernel_;
        global = global_;
    }
    const cl::Context& getContext() const {
        return context;
    }

    const cl::CommandQueue& getQueue() const {
        return queue;
    }

    cl::Kernel& getKernel() const {
        return kernel;
    }

    const cl::NDRange& getGlobal() const {
        return global;
    }
    clStruct& operator=(const clStruct& right) {
        //проверка на самоприсваивание
        if (this == &right) {
            return *this;
        }
        context = right.context;
        queue = right.queue;
        kernel = right.kernel;
        global = right.global;
        return *this;
    }
};

clStruct mainFunction(std::string kernelFileName, std::vector<int>& layerSizes, std::vector<float>& weights,
        const char* functionName, std::vector<int>& counters){
	std::vector<cl::Platform> allPlatforms;
    cl::Platform::get(&allPlatforms);
    if (allPlatforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

	cl::Platform defaultPlatform = allPlatforms[0];
    std::vector<cl::Device> allDevices;
    defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
    if (allDevices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device defaultDevice = allDevices[0];
    cl::Context context(defaultDevice);

	std::ifstream sourceFile(kernelFileName);
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));;
    cl::Program program = cl::Program(context, source);
    if (program.build({defaultDevice}) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice) << "\n";
        getchar();
        exit(1);
    }


    cl::Buffer layerSizesBuffer(context, CL_MEM_READ_WRITE, layerSizes.size()* sizeof(int));
    cl::Buffer weightsBuffer(context, CL_MEM_READ_WRITE, weights.size()* sizeof(float));
    cl::Buffer countersBuffer(context, CL_MEM_READ_WRITE, counters.size()* sizeof(int));

    cl::CommandQueue queue(context, defaultDevice);
    queue.enqueueWriteBuffer(layerSizesBuffer, CL_TRUE, 0, layerSizes.size() * sizeof(int), layerSizes.data());
    queue.enqueueWriteBuffer(weightsBuffer, CL_TRUE, 0, weights.size() * sizeof(float), weights.data());
    queue.enqueueWriteBuffer(countersBuffer, CL_TRUE, 0, counters.size() * sizeof(int), counters.data());

    cl::Kernel kernel(program, functionName);

    kernel.setArg(1, layerSizesBuffer);
    kernel.setArg(2, weightsBuffer);
    kernel.setArg(6, countersBuffer);

    size_t threadNumber = 0;
    for (int i = 1; i < layerSizes.size(); ++i) {
        threadNumber += layerSizes[i];
    }

    cl::NDRange global(threadNumber);
    clStruct clStructVariables(context, queue, kernel, global);
    return clStructVariables;
}

void loopFunction(clStruct clStructVariables,std::vector<float>& values, std::vector<float>& input, std::vector<float>& output){
    cl::Buffer valuesBuffer(clStructVariables.getContext(), CL_MEM_READ_WRITE, values.size() * sizeof(float));
    cl::Buffer inputBuffer(clStructVariables.getContext(), CL_MEM_READ_WRITE, input.size() * sizeof(float));
    cl::Buffer outputBuffer(clStructVariables.getContext(), CL_MEM_READ_WRITE, output.size() * sizeof(float));

    clStructVariables.getQueue().enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, input.size() * sizeof(float), input.data());

    clStructVariables.getKernel().setArg(3, valuesBuffer);
    clStructVariables.getKernel().setArg(4, inputBuffer);
    clStructVariables.getKernel().setArg(5, outputBuffer);

    clStructVariables.getQueue().enqueueNDRangeKernel(clStructVariables.getKernel(), cl::NullRange, clStructVariables.getGlobal());
    clStructVariables.getQueue().finish();

    clStructVariables.getQueue().enqueueReadBuffer(outputBuffer, CL_TRUE, 0, output.size() * sizeof(float), output.data());
}

int main() {
    std::vector<int> layerSizes;
    std::vector<float> weights;
    std::vector<float> values;
    std::vector<float> biases;
    std::vector<float> input;
    std::vector<float> output;
    std::vector<int> counters;
  
    counters = layerSizes;
    std::string kernelFileName = "neuron.cl";
    char array[] = "neuron";
    const char* functionName = array;
    clStruct clStructValues = mainFunction( kernelFileName, layerSizes, weights, functionName, counters);
    loopFunction(clStructValues,  values, input, output);
    return 0;
}