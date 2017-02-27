#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <iterator>

#define __CL_ENABLE_EXCEPTIONS

#include <cl.hpp>

const std::size_t N = 10;
const std::string fileName = "kernel.cl";

int main() {

    try {
        std::vector<cl::Platform> availablePlatforms;
        cl::Platform::get(&availablePlatforms);
        if (availablePlatforms.size() == 0) {
            std::cout << "No available platforms." << std::endl;
            return EXIT_FAILURE;
        }

        cl_context_properties properties[] =
                {CL_CONTEXT_PLATFORM, (cl_context_properties)(availablePlatforms[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_DEFAULT, properties);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        cl::CommandQueue queue(context, devices[0]);

        std::ifstream file(fileName);
        std::string fileContent(std::istreambuf_iterator<char>(file), {});

        cl::Program::Sources kernelSources{std::make_pair(fileContent.c_str(), fileContent.length())};
        cl::Program program(context, kernelSources);
        program.build(devices);

        cl::Kernel kernel(program, "kernel1");

        auto memory = std::make_unique<cl_int[]>(N);
        cl::Buffer buffer(context, CL_MEM_READ_WRITE, N * sizeof(cl_int));
        queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, N * sizeof(cl_int), memory.get());
        kernel.setArg(0, buffer);

        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NullRange, nullptr, &event);
        queue.enqueueReadBuffer(buffer, CL_TRUE, 0, N * sizeof(cl_int), memory.get());

        for (std::size_t i = 0; i < N; ++i) {
            std::cout << memory.get()[i] << ' ';
        }

        event.wait();
    }
    catch (cl::Error& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}