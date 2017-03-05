#include <cl.hpp>
#include <iostream>
#include <fstream>
#include <memory>
#include <iomanip>

#define __CL_ENABLE_EXCEPTIONS

static constexpr bool gDebugEnabled = false;
#define debugStream\
    if (!gDebugEnabled) {}\
    else std::cout

std::tuple<cl::Kernel, cl::Context, cl::CommandQueue> get_kernel_and_shit(const std::string& path, const std::string& name)
{
    // Find devices
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    debugStream << "Found " << platforms.size() << " platforms\n";

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    debugStream << "Found " << devices.size() << " devices for first platform\n";

    std::vector<cl::Device> contextDevices;
    contextDevices.push_back(devices[0]);
    cl::Context context(contextDevices);
    cl::CommandQueue queue(context, devices[0]);

    // Load kernel source
    std::ifstream sourceFile(path);
    std::string sourceStr(std::istreambuf_iterator<char>(sourceFile),(std::istreambuf_iterator<char>()));
    debugStream << "Kernel source:\n" << sourceStr;

    // Build kernel
    cl::Program program(context, cl::Program::Sources(1, std::make_pair(sourceStr.c_str(), sourceStr.size() + 1)));
    debugStream << "Program built with code " << program.build(contextDevices) << "\n";
    cl::Kernel kernel(program, name.c_str());
    return std::make_tuple(kernel, context, queue);
}

void print_matrix(int* a, size_t size)
{
    debugStream << "Matrix: \n";
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
        {
            debugStream << std::setfill(' ') << std::setw(12) << a[i * size + j];
        }
        debugStream << '\n';
    }
}

int main(void)
{
    cl::Kernel kernel;
    cl::Context context;
    cl::CommandQueue queue;
    std::tie(kernel, context, queue) = get_kernel_and_shit("kernel.cl", "SimpleKernel");

    std::cout << "Enter size: ";
    size_t size;
    std::cin >> size;
    std::unique_ptr<int[]> res(new int[size]);
    cl::Buffer clBuf = cl::Buffer(context, CL_MEM_READ_WRITE, size * sizeof(int));

    kernel.setArg(0, clBuf);
    kernel.setArg(1, static_cast<unsigned int>(size));

    // Run kernel
    debugStream << "Run status: " << queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange) << '\n';
    queue.finish();
    queue.enqueueReadBuffer(clBuf, CL_TRUE, 0, size * sizeof(int), res.get());

    // Print result
    for (size_t i = 0; i < size; ++i)
    {
        debugStream << res[i] << ' ';
    }

    return 0;
}