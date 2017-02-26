#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>

#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

int main() {

    const std::string fileName = "kernel.cl";

    cl_int err = CL_SUCCESS;
    try
    {
        std::vector<cl::Platform> availablePlatforms;
        cl::Platform::get(&availablePlatforms);

        cl_context_properties properties[] =
                {CL_CONTEXT_PLATFORM, (cl_context_properties) (availablePlatforms[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_DEFAULT, properties); // ?
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        std::ifstream file(fileName);
        std::stringstream fileBuffer;
        fileBuffer << file.rdbuf();

        cl::Program::Sources kernelSources{std::make_pair(fileBuffer.str().c_str(), fileBuffer.str().length())};
        cl::Program program(context, kernelSources);
        program.build(devices);

        cl::Kernel kernel(program, "task1");

        cl::CommandQueue queue(context, devices[0]);

        const int numElements = 10;
        auto memory = std::make_unique<cl_int[]>(numElements);
        cl::Buffer buffer(context, CL_MEM_READ_WRITE, numElements * sizeof(cl_int));
        queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, numElements * sizeof(cl_int), memory.get());
        kernel.setArg(0, buffer);

        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(4, 4), cl::NullRange);

        queue.enqueueReadBuffer(buffer, CL_TRUE, 0, numElements * sizeof(cl_int), memory.get());

        event.wait();
    }
    catch (cl::Error error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
    }

    return EXIT_SUCCESS;
}