#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cassert>
#include <chrono>

#define __CL_ENABLE_EXCEPTIONS

#include <cl.hpp>

using namespace std;

string loadSourceCode(string filename) {
    ifstream sourceFile;
    sourceFile.open(filename);
    assert(sourceFile.is_open());
    return string((istreambuf_iterator<char>(sourceFile)), istreambuf_iterator<char>());
}

const string sourceCode = loadSourceCode("kernel.cl");

int main() {
    unsigned N = 1000;
    unsigned matrix_size = N * N;
    try {
        vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        if (all_platforms.size() == 0) {
            cout << " No platforms found. Check OpenCL installation!\n";
            exit(1);
        }
        cl::Platform platform = all_platforms[0];
        cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

        vector<cl::Device> all_devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        if (all_devices.size() == 0) {
            cout << " No devices found. Check OpenCL installation!\n";
            exit(1);
        }
        cl::Device device = all_devices[0];
        cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

        cl::Context context(device);

        cl::CommandQueue queue(context, device);

        cl::Program::Sources sources{{sourceCode.c_str(), sourceCode.size()}};

        cl::Program program(context, sources);

        try {
            program.build({device});
        }
        catch (cl::Error &error) {
            string build_log;
            program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &build_log);
            throw build_log;
        }
        // reduced arrays
        vector<cl_int> A(matrix_size);
        vector<cl_int> B(matrix_size);
        vector<cl_int> C(matrix_size);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i * N + j] = rand() % 10;
                B[i * N + j] = rand() % 10;
            }
        }

        auto start = chrono::steady_clock::now();
        cl::Buffer bufferA(context, CL_MEM_READ_WRITE, A.size() * sizeof(cl_int));
        cl::Buffer bufferB(context, CL_MEM_READ_WRITE, B.size() * sizeof(cl_int));
        cl::Buffer bufferC(context, CL_MEM_READ_WRITE, C.size() * sizeof(cl_int));
        cl::copy(queue, A.begin(), A.end(), bufferA);
        cl::copy(queue, B.begin(), B.end(), bufferB);

        // create kernel functor
        auto kernel = cl::make_kernel<const cl::Buffer &, const cl::Buffer &, cl::Buffer &, cl_int>(program, "job");
        kernel(cl::EnqueueArgs(queue, cl::NDRange(N)), bufferA, bufferB, bufferC, N).wait();

        cl::copy(queue, bufferC, C.begin(), C.end());
        cout << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << endl;
    }
    catch (cl::Error &error) {
        cerr << "ERROR: " << error.what() << "(" << error.err() << ")" << endl;
        return EXIT_FAILURE;
    }
    catch (string &log) {
        cerr << log << endl;
        return EXIT_FAILURE;
    }
    return 0;
}