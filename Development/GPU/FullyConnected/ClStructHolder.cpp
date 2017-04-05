#include "ClStructHolder.h"

ClStructHolder::ClStructHolder(cl::Context context, cl::CommandQueue queue, cl::Kernel kernel, size_t threadNumber,
        size_t groupSize) :
        context(context),
        queue(queue),
        kernel(kernel),
        globalRange(threadNumber),
        localRange(groupSize) {
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

const cl::NDRange& ClStructHolder::getGlobalRange() const {
    return globalRange;
}

const cl::NDRange& ClStructHolder::getLocalRange() const {
    return localRange;
}
