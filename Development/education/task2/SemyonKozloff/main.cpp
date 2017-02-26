#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <iterator>

#include <cl.hpp>
#include <random>

#define __CL_ENABLE_EXCEPTIONS

const std::size_t N = 5;

void fillMatrix(cl_int** matrix, std::size_t n) {
    std::random_device randomDevice;
    std::default_random_engine engine(randomDevice());
    std::uniform_int_distribution<cl_int> uniformDistribution(1, 100);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            matrix[i][j] = uniformDistribution(engine) % 10;
        }
    }
}

void printMatrix(cl_int** matrix, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            std::cout << matrix[i][j] << ' ';
        }
        std::cout << std::endl;
    }
}

int main() {

    const std::string fileName = "kernel.cl";

    try {
        std::vector<cl::Platform> availablePlatforms;
        cl::Platform::get(&availablePlatforms);
        if (availablePlatforms.size() == 0) {
            std::cout << "No available platforms." << std::endl;
            return EXIT_FAILURE;
        }

        cl_context_properties properties[] =
                {CL_CONTEXT_PLATFORM, (cl_context_properties)(availablePlatforms[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties); // add properties?
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        cl::CommandQueue queue(context, devices[0]);

        std::ifstream file(fileName);
        std::string fileContent(std::istreambuf_iterator<char>(file), {});

        cl::Program::Sources kernelSources{std::make_pair(fileContent.c_str(), fileContent.length())};
        cl::Program program(context, kernelSources);
        program.build(devices);

        cl::Kernel kernel(program, "multiplier");

        auto a = new cl_int*[N];
        auto b = new cl_int*[N];
        auto c = new cl_int*[N];
        for (std::size_t i = 0; i < N; ++i) {
            a[i] = new cl_int[N];
            b[i] = new cl_int[N];
            c[i] = new cl_int[N];
        }
        ::fillMatrix(a, N);
        ::fillMatrix(b, N);

        ::printMatrix(a, N);
        std::cout << 'X' << std::endl;
        ::printMatrix(b, N);
        std::cout << '=' << std::endl;

        std::size_t matrixMemorySize = N * N * sizeof(cl_int);

        cl::Buffer aBuffer(context, CL_MEM_READ_ONLY, matrixMemorySize);
        cl::Buffer bBuffer(context, CL_MEM_READ_ONLY, matrixMemorySize);
        cl::Buffer cBuffer(context, CL_MEM_WRITE_ONLY, matrixMemorySize);
        cl::Buffer nBuffer(context, CL_MEM_READ_ONLY, sizeof(std::size_t));
        queue.enqueueWriteBuffer(aBuffer, CL_TRUE, 0, matrixMemorySize, a);
        queue.enqueueWriteBuffer(bBuffer, CL_TRUE, 0, matrixMemorySize, b);
        queue.enqueueWriteBuffer(cBuffer, CL_TRUE, 0, matrixMemorySize, c);
        queue.enqueueWriteBuffer(nBuffer, CL_TRUE, 0, sizeof(std::size_t), &N);
        kernel.setArg(0, aBuffer);
        kernel.setArg(1, bBuffer);
        kernel.setArg(2, cBuffer);
        kernel.setArg(3, nBuffer);

        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NullRange, nullptr, &event);
        queue.enqueueReadBuffer(cBuffer, CL_TRUE, 0, matrixMemorySize, c);

        event.wait();

        ::printMatrix(c, N);

        for (std::size_t i = 0; i < N; ++i) {
            delete a[i];
            delete b[i];
            delete c[i];
        }
        delete a;
        delete b;
        delete c;

    }
    catch (cl_int err) {
        std::cerr << "ERROR: " << err << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}