#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>

#define __CL_ENABLE_EXCEPTIONS

#include "../include/cl.hpp"
#include "../include/err_code.h"

using namespace std;

void printError(cl::Error err) {
    std::cout << "Exception\n";
    std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
}

std::unique_ptr<int[]> performTaskOnDevice(cl::Device device, int n, std::string filename) {
    try {
        std::vector<cl::Device> contextDevices;
        contextDevices.push_back(device);
        cl::Context context(contextDevices);

        cl::CommandQueue queue(context, device);

        std::ifstream sourceFile(filename);
        std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
        cl::Program program(context, sourceCode, true);
        cl::Kernel kernel(program, "arrayIndeces");

        cl::Buffer clmOutputVector = cl::Buffer(context,
            CL_MEM_READ_WRITE, n * sizeof(int));
        kernel.setArg(0, clmOutputVector);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);
        queue.finish();

        auto pOutputVector = std::make_unique<int[]>(n);
        queue.enqueueReadBuffer(clmOutputVector, CL_TRUE, 0, n * sizeof(int), pOutputVector.get());

        return std::move(pOutputVector);
    } catch (cl::Error err) {
        printError(err);
    }
}

cl::Device getDevice() {
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            throw std::logic_error("Platforms size 0");
        }

        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.size() == 0) {
            throw std::logic_error("Devices size 0");
        }
        return devices[0];
    } catch (cl::Error err) {
        printError(err);
    }
}

class Matrix {
  private:
    vector<vector<int>> matrix;
  public:
    Matrix(int n, int m) {
        srand(43);
        for (int i = 0; i < n; i++) {
            matrix.push_back(vector<int>(m));
            for (int j = 0; j < m; j++) {
                if (random) {
                    matrix[i][j] = rand() % 100;
                }
            }
        }
    }
};

int main(int argv, char** argc) {
    int n;
    std::string filename = "matrixMultiplication.cl";
    std::cin >> n;

    try {
        cl::Device device = getDevice();

        std::unique_ptr<int[]> ans = performTaskOnDevice(device, n, filename);
        for (int i = 0; i < n; i++) {
            std::cout << ans.get()[i] << " ";
        }
    } catch (cl::Error err) {
        printError(err);
    }
    return EXIT_SUCCESS;
}

