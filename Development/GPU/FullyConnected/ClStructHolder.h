#pragma once

#include <cl.hpp>

class ClStructHolder {
private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel kernel;
    cl::NDRange globalRange;
    cl::NDRange localRange;

public:
    ClStructHolder(cl::Context context, cl::CommandQueue queue, cl::Kernel kernel, size_t threadNumber, size_t groupSize);

    const cl::Context& getContext() const;
    const cl::CommandQueue& getQueue() const;
    cl::Kernel& getKernel();
    const cl::NDRange& getGlobalRange() const;
    const cl::NDRange& getLocalRange() const;
};
