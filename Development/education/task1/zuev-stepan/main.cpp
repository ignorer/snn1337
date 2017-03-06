#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <memory>
#include <iomanip>

#include <cl.hpp>

class NullStream
{
public:
    template <typename T>
    NullStream& operator <<(const T& t)
    {
        return *this;
    }
};

#ifdef DEBUG
#define debugOut std::cout
#else
auto debugOut = NullStream();
#endif

std::tuple<cl::Kernel, cl::Context, cl::CommandQueue> getKernel(const std::string& path, const std::string& name)
{
    // Find devices
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    debugOut << "Found " << platforms.size() << " platforms\n";

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    debugOut << "Found " << devices.size() << " devices for first platform\n";

    std::vector<cl::Device> contextDevices;
    contextDevices.push_back(devices[0]);
    cl::Context context(contextDevices);
    cl::CommandQueue queue(context, devices[0]);

    // Load kernel source
    std::ifstream sourceFile(path);
    std::string sourceStr(std::istreambuf_iterator<char>(sourceFile),(std::istreambuf_iterator<char>()));
    debugOut << "Kernel source:\n" << sourceStr;

    // Build kernel
    cl::Program program(context, cl::Program::Sources(1, std::make_pair(sourceStr.c_str(), sourceStr.size() + 1)));
    debugOut << "Program built with code " << program.build(contextDevices) << "\n";
    cl::Kernel kernel(program, name.c_str());
    return std::make_tuple(kernel, context, queue);
}

int main(void)
{
    cl::Kernel kernel;
    cl::Context context;
    cl::CommandQueue queue;
    std::tie(kernel, context, queue) = getKernel("kernel.cl", "SimpleKernel");

    std::cout << "Enter size: ";
    size_t size;
    std::cin >> size;
    std::unique_ptr<int[]> res(new int[size]);
    cl::Buffer clBuf = cl::Buffer(context, CL_MEM_READ_WRITE, size * sizeof(int));

    kernel.setArg(0, clBuf);
    kernel.setArg(1, static_cast<unsigned int>(size));

    // Run kernel
    debugOut << "Run: " << queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange) << '\n';
    queue.finish();
    queue.enqueueReadBuffer(clBuf, CL_TRUE, 0, size * sizeof(int), res.get());

    // Print result
    for (size_t i = 0; i < size; ++i)
    {
        debugOut << res[i] << ' ';
    }

    return 0;
}