#pragma once

#include <cl.hpp>

class ClStructHolder {
private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel kernel;
    cl::NDRange globalRange;

public:
    ClStructHolder(cl::Context context, cl::CommandQueue queue, cl::Kernel kernel, size_t threadNumber);

    const cl::Context& getContext() const;
    const cl::CommandQueue& getQueue() const;
    cl::Kernel& getKernel();
    const cl::NDRange& getGlobalRange() const;
};
