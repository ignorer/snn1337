#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <iterator>
#include <random>

#define __CL_ENABLE_EXCEPTIONS

#include <cl.hpp>

const std::size_t N = 5;
const std::string fileName = "kernel.cl";

void fillMatrix(cl_int matrix[N * N]) {
    std::random_device randomDevice;
	std::default_random_engine engine(randomDevice());
	std::uniform_int_distribution<cl_int> uniformDistribution(1, 100);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            matrix[i * N + j] = uniformDistribution(engine) % 10;
        }
    }
}

void fillMatrix(cl_int matrix[N * N], cl_int value) {
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            matrix[i * N + j] = value;
        }
    }
}

void printMatrix(cl_int matrix[N * N]) {
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            std::cout << matrix[i * N + j] << ' ';
        }
        std::cout << std::endl;
    }
}

int main() {

	cl_int a[N * N], b[N * N], c[N * N];

	::fillMatrix(a);
	::fillMatrix(b);
	::fillMatrix(c, 0);

	::printMatrix(a);
	std::cout << 'X' << std::endl;
	::printMatrix(b);
	std::cout << '=' << std::endl;

	cl::STRING_CLASS buildInfoLog;

	try {
        std::vector<cl::Platform> availablePlatforms;
        cl::Platform::get(&availablePlatforms);
        if (availablePlatforms.size() == 0) {
            throw cl::Error(1, "No available platforms.");
        }

        cl_context_properties properties[] =
                {CL_CONTEXT_PLATFORM, (cl_context_properties)(availablePlatforms[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        cl::CommandQueue queue(context, devices[0]);

        std::ifstream file(fileName);
        std::string fileContent(std::istreambuf_iterator<char>(file), {});

        cl::Program::Sources kernelSources{std::make_pair(fileContent.c_str(), fileContent.length())};
        cl::Program program(context, kernelSources);

		try {
			program.build(devices);
		}
		catch (cl::Error& error) {
			program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &buildInfoLog);
			throw cl::Error(error.err(), buildInfoLog.c_str());
		}

        cl::Kernel kernel(program, "multiplier");

        std::size_t matrixMemorySize = N * N * sizeof(cl_int);

        cl::Buffer aBuffer(context, CL_MEM_READ_ONLY, matrixMemorySize);
        cl::Buffer bBuffer(context, CL_MEM_READ_ONLY, matrixMemorySize);
        cl::Buffer cBuffer(context, CL_MEM_READ_WRITE, matrixMemorySize);
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

        ::printMatrix(c);
    }
    catch (cl::Error& error) {
        std::cerr << "ERROR: " << error.err() << std::endl;
	    std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}