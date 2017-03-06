#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "err_code.h"

using namespace std;
using namespace chrono;

cl::Device getRandomDevice(int type = CL_DEVICE_TYPE_ALL) {
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

int main() {
    size_t n = 10;
    vector<int> A(n, 0);

    try {
        // prepare
        auto gpu = getRandomDevice(CL_DEVICE_TYPE_GPU);
        cl::Context context(gpu);
        cl::CommandQueue commandQueue(context, gpu);

        // compile kernel
        ifstream stream("ignorerKernel.cl");
        cl::Program program(context, string(istreambuf_iterator<char>(stream), {}), true);
        cl::make_kernel<cl::Buffer> kernel(program, "fillArray");

        // run
        cl::Buffer buffer(context, A.begin(), A.end(), true);
        kernel(cl::EnqueueArgs(commandQueue, cl::NDRange(n)), buffer);
        cl::copy(commandQueue, buffer, A.begin(), A.end());

        // check
        for_each(A.begin(), A.end(), [](auto x){cout << x << " ";});
    } catch (const cl::Error& e) {
        cout << string(e.what()) + ": " + err_code(e.err()) << endl;
    }
}