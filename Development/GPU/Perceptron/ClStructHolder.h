#pragma once

#include <string>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS

#include <cl.hpp>

class ClStructHolder {
private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel kernel;

    cl::Device getDevice(cl_device_type deviceType);

public:
    ClStructHolder(const std::string& kernelFileName, const std::vector<int>& layerSizes, const std::vector<float>& weights,
            const std::string& functionName, cl_device_type deviceType);

    const cl::Context& getContext() const;
    const cl::CommandQueue& getQueue() const;
    cl::Kernel& getKernel();
};
