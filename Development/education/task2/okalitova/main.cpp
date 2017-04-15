#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>

#define __CL_ENABLE_EXCEPTIONS

#include <cl.hpp>
#include <err_code.h>

using namespace std;

void printError(cl::Error err) {
    std::cout << "Exception\n";
    std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
}

class Matrix {
private:
    int n, m;
    vector<vector<int>> matrix;
public:
    Matrix(int n, int m) {
        this->n = n;
        this->m = m;
        for (int i = 0; i < n; i++) {
            matrix.push_back(vector<int>(m));
            for (int j = 0; j < m; j++) {
                if (random) {
                    matrix[i][j] = rand() % 100;
                }
            }
        }
    }

    vector<int> getPlainMatrix() {
        vector<int> plainMatrix;
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix[i].size(); j++) {
                plainMatrix.push_back(matrix[i][j]);
            }
        }
        return plainMatrix;
    }

    int getN() const {
        return n;
    }

    int getM() const {
        return m;
    }

    void print() {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                cout << matrix[i][j] << " ";
            }
            cout << "\n";
        }
    }
};

std::unique_ptr<int[]> performTaskOnDevice(cl::Device device, std::string filename,
    Matrix A, Matrix B) {
    try {
        std::vector<cl::Device> contextDevices;
        contextDevices.push_back(device);
        cl::Context context(contextDevices);

        cl::CommandQueue queue(context, device);

        std::ifstream sourceFile(filename);
        std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
        cl::Program program(context, sourceCode, true);
        cl::Kernel kernel(program, "matrixMultiplication");

        vector<int> a = A.getPlainMatrix();
        vector<int> b = B.getPlainMatrix();

        cl::Buffer clmInputVector1 = cl::Buffer(context,
            CL_MEM_READ_ONLY, a.size() * sizeof(int));
        cl::Buffer clmInputVector2 = cl::Buffer(context,
            CL_MEM_READ_ONLY, b.size() * sizeof(int));
        cl::Buffer clmOutputVector = cl::Buffer(context,
            CL_MEM_READ_WRITE, A.getN() * B.getM() * sizeof(int));
        cl::copy(queue, a.begin(), a.end(), clmInputVector1);
        cl::copy(queue, b.begin(), b.end(), clmInputVector2);

        kernel.setArg(0, clmInputVector1);
        kernel.setArg(1, clmInputVector2);
        kernel.setArg(2, clmOutputVector);
        kernel.setArg(3, A.getN());
        kernel.setArg(4, A.getM());
        kernel.setArg(5, B.getM());

        int workDim = 2;
        int globalWorkSize[workDim] = {A.getN(), B.getM()};
        queue.enqueueNDRangeKernel(kernel,
            cl::NullRange, cl::NDRange(globalWorkSize[0], globalWorkSize[1]), cl::NullRange);

        queue.finish();

        auto pOutputVector = std::make_unique<int[]>(A.getN() * B.getM());
        queue.enqueueReadBuffer(clmOutputVector, CL_TRUE, 0,
            A.getN() * B.getM() * sizeof(int), pOutputVector.get());

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

int main(int argv, char** argc) {
    srand(43);
    std::string filename = "matrixMultiplication.cl";

    try {
        cl::Device device = getDevice();

        Matrix A(2, 3);
        Matrix B(3, 4);

        std::unique_ptr<int[]> ans = performTaskOnDevice(device,filename, A, B);
        for (int i = 0; i < A.getN(); i++) {
            for (int j = 0; j < B.getM(); j++) {
                std::cout << ans.get()[i * B.getM() + j] << " ";
            }
            cout << "\n";
        }
    } catch (cl::Error err) {
        printError(err);
    }
    return EXIT_SUCCESS;
}

