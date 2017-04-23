//
// Created by mariia on 22.04.2017.
//

#ifndef GPU_CLSTRUCTHOLDER_H
#define GPU_CLSTRUCTHOLDER_H

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
#endif //GPU_CLSTRUCTHOLDER_H
