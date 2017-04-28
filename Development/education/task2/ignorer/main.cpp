#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "err_code.h"

using namespace std;
using namespace chrono;

cl::Device getRandomDevice(cl_device_type type = CL_DEVICE_TYPE_ALL) {
    vector<cl::Device> devices;
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto& platform : platforms) {
        try {
            platform.getDevices(type, &devices);
            if (!devices.empty()) {
                return {devices.front()};
            }
        } catch (const cl::Error& e) {
            // ignore this stuff
        }
    }
    throw std::runtime_error("device not found");
}

void printMatrix(const vector<int> M, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << M[i * n + j] << " ";
        }
        cout << endl;
    }
}

int main() {
    // generate test matrices
    size_t n = 1000;
    vector<int> A(n * n);
    vector<int> B(n * n);
    vector<int> C(n * n);
    for (int i = 0; i < n * n; ++i) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }
//        printMatrix(A, n);
//        printMatrix(B, n);

    try {
        // prepare
        auto gpu = getRandomDevice(CL_DEVICE_TYPE_CPU); // CPU solves this task faster
        cl::Context context(gpu);
        cl::CommandQueue commandQueue(context, gpu);

        // read and compile kernel
        ifstream stream("ignorerKernel.cl");
        cl::Program program(context, string(istreambuf_iterator<char>(stream), {}), true);
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, unsigned int> kernel(program, "multiply");

        auto start = steady_clock::now(); // start timer
        // transpose second matrix to make it work faster (use cache lines)
        vector<int> B2(n * n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                B2[i * n + j] = B[i + j * n];
            }
        }

        // run
        cl::Buffer bufferA(context, A.begin(), A.end(), true);
        cl::Buffer bufferB(context, B2.begin(), B2.end(), true);
        cl::Buffer bufferC(context, C.begin(), C.end(), true);
        kernel(cl::EnqueueArgs(commandQueue, cl::NDRange(n)), bufferA, bufferB, bufferC, n);
        cl::copy(commandQueue, bufferC, C.begin(), C.end());

        cout << duration_cast<milliseconds>(steady_clock::now() - start).count() << endl;

//        printMatrix(C, n);
    } catch (const cl::Error& e) {
        cout << string(e.what()) + ": " + err_code(e.err()) << endl;
    }
}