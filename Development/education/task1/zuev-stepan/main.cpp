#include <cl.hpp>
#include <iostream>
#include <fstream>
#include <bits/unique_ptr.h>

#define __CL_ENABLE_EXCEPTIONS

int main(void)
{
    // Find devices
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::cerr << "Found " << platforms.size() << " platforms\n";

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    std::cerr << "Found " << devices.size() << " devices for first platform\n";

    std::vector<cl::Device> contextDevices;
    contextDevices.push_back(devices[0]);
    cl::Context context(contextDevices);
    cl::CommandQueue queue(context, devices[0]);

    // Load kernel
    std::ifstream sourceFile("kernel.cl");
    std::string sourceStr(std::istreambuf_iterator<char>(sourceFile),(std::istreambuf_iterator<char>()));
    std::cerr << "Kernel source:\n" << sourceStr;

    // Build kernel
    cl::Program::Sources source(1, std::make_pair(sourceStr.c_str(), sourceStr.size() + 1));

    cl::Program program(context, source);
    std::cerr << "Program built with code " << program.build(contextDevices) << "\n";
    cl::Kernel kernel(program, "SimpleKernel");

    // Set args
    std::cout << "Enter size: ";
    size_t size;
    std::cin >> size;
    std::unique_ptr<int[]> res(new int[size]);
    cl::Buffer clBuf = cl::Buffer(context, CL_MEM_READ_WRITE, size * sizeof(int));

    kernel.setArg(0, clBuf);
    kernel.setArg(1, static_cast<unsigned int>(size));

    // Run kernel
    std::cerr << "Run status: "
              << queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange) << '\n';
    queue.finish();
    queue.enqueueReadBuffer(clBuf, CL_TRUE, 0, size * sizeof(int), res.get());

    // Print result
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << res[i] << ' ';
    }

    return 0;
}