#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <iterator>
#include <cassert>

#define __CL_ENABLE_EXCEPTIONS

#include <cl.hpp>


std::string readFileAsString(std::string fileName) {
    std::ifstream ifs(fileName);
    assert(ifs.is_open());
    return std::string(std::istreambuf_iterator<char>(ifs),
            std::istreambuf_iterator<char>());
    
}

const std::string kernelEnumerateString = readFileAsString("enumerate.cl");
const std::pair<const char *, size_t>
        kernelEnumerateSource(kernelEnumerateString.c_str(),
                kernelEnumerateString.size());


int main(int argc, char **argv) {
    
    int N = 5;
    if (argc >= 2) {
        sscanf(argv[1], "%d", &N);
    }
    
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << "No available platforms." << std::endl;
            return EXIT_FAILURE;
        } else {
            std::cout << "Available platforms:" << std::endl;
            for (cl::Platform platform : platforms) {
                std::cout << "\t" << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
                
            }
        }
        
        cl::Platform platform = platforms.front();
        std::cout << "Chosen platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        
        if (devices.size() == 0) {
            std::cout << "No available devices." << std::endl;
            return EXIT_FAILURE;
        } else {
            std::cout << "Available devices:" << std::endl;
            for (cl::Device device : devices) {
                std::cout << "\t" << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            }
        }
        
        cl::Device device = std::move(devices.front());
        devices.clear();
        
        std::cout << "Chosen device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        
        cl::Context context(device);
        
        cl::CommandQueue queue(context, device);
        
        cl::Program::Sources kernelSources{kernelEnumerateSource};
        
        cl::Program program(context, kernelSources);
        program.build({device});
        
        cl::Kernel kernel(program, "kernel_enumerate");
        
        std::vector<cl_int> memoryData(N);
        size_t memoryDataSizeInBytes = memoryData.size() * sizeof(cl_int);
        cl::Buffer buffer(context, CL_MEM_READ_WRITE, memoryDataSizeInBytes);
        queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, memoryDataSizeInBytes, memoryData.data());
        
        kernel.setArg(0, buffer);
        
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NullRange, nullptr, &event);
        queue.enqueueReadBuffer(buffer, CL_TRUE, 0, memoryDataSizeInBytes, memoryData.data());
        event.wait();
        
        for (cl_int x : memoryData) {
            std::cout << x << ' ';
        }
        
    }
    catch (cl::Error &error) {
        std::cerr << "ERROR: " << error.what() << "(" << error.err() << ")" << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}