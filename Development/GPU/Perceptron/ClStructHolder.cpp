#include "ClStructHolder.h"

#include <fstream>
#include <numeric>

using namespace std;

cl::Device ClStructHolder::getDevice(cl_device_type deviceType) {
    vector<cl::Platform> allPlatforms;
    cl::Platform::get(&allPlatforms);
    vector<cl::Device> allDevices;
    for (auto& platform : allPlatforms) {
        try {
            platform.getDevices(CL_DEVICE_TYPE_GPU, &allDevices);
            if (!allDevices.empty()) {
                break;
            }
        } catch (const exception& e) {
            // ignore that guy
        }
    }
    return allDevices[0];
}

ClStructHolder::ClStructHolder(const string& kernelFileName, const vector<int>& layerSizes,
        const vector<float>& weights, const string& functionName, cl_device_type deviceType) {
    cl::Device device = getDevice(CL_DEVICE_TYPE_GPU);
    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);

    ifstream sourceFile(kernelFileName);
    string sourceCode(istreambuf_iterator<char>(sourceFile), (istreambuf_iterator<char>()));
    cl::Program::Sources source(1, make_pair(sourceCode.c_str(), sourceCode.length() + 1));
    cl::Program program = cl::Program(context, source);

    try {
        program.build({device});
    }
    catch (cl::Error& error) {
        string buildLog;
        program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &buildLog);
        throw runtime_error(buildLog);
    }

    kernel = cl::Kernel(program, functionName.c_str());
}

const cl::Context& ClStructHolder::getContext() const {
    return context;
}

const cl::CommandQueue& ClStructHolder::getQueue() const {
    return queue;
}

cl::Kernel& ClStructHolder::getKernel() {
    return kernel;
}
