#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <iterator>
#include <cassert>
#include <chrono>
#include <ctime>

#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif


std::string readFileAsString(std::string fileName) {
	std::ifstream ifs(fileName);
	assert(ifs.is_open());
	return std::string(std::istreambuf_iterator<char>(ifs),
	                   std::istreambuf_iterator<char>());
}

const std::string kernelMultiplyName = "kernel_multiply";
const std::string kernelMultiplyString = readFileAsString("../multiply.cl");
const std::pair<const char *, size_t>
		kernelMultiplySource(kernelMultiplyString.c_str(),
		             kernelMultiplyString.size());

std::vector<cl_int> readMatrixRaw(std::string fileName) {
	FILE *input = fopen(fileName.c_str(), "rt");
	assert(input);
	int N, M;
	fscanf(input, "%d%d", &N, &M);
	std::vector<cl_int> raw((size_t)2 + N * M);
	raw[0] = N;
	raw[1] = M;
	for (int i = 0; i < N * M; i++)
		fscanf(input, "%d", &raw[2 + i]);
	fclose(input);
	return raw;
}

void writeMatrixRaw(std::string fileName, const std::vector<cl_int> &raw) {
	FILE *output = fopen(fileName.c_str(), "wt");
	assert(output);
	int N = raw[0], M = raw[1];
	fprintf(output, "%d %d\n", N, M);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			fprintf(output, "%d ", raw[2 + i * M + j]);
		}
		fprintf(output, "\n");
	}
	fclose(output);
}

std::vector<cl_int> transposeMatrixRaw(const std::vector<cl_int> &rawMatrix) {
	std::vector<cl_int> transposed(rawMatrix.size());
	int N = rawMatrix[0], M = rawMatrix[1];
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			transposed[2 + j * N + i] = rawMatrix[2 + i * M + j];
	transposed[0] = M;
	transposed[1] = N;
	return transposed;
}

std::vector<cl_int> concatenate(const std::vector<cl_int> &a, const std::vector<cl_int> &b) {
	std::vector<cl_int> result;
	result.reserve(a.size() + b.size());
	result.insert(result.end(), a.begin(), a.end());
	result.insert(result.end(), b.begin(), b.end());
	return result;
}


int main(int argc, char **argv) {
	int resultN, resultM;
	std::vector<cl_int> rawMatrixA, rawMatrixB, rawMatrixBTransposed, rawResult;
	if (argc >= 4) {
		rawMatrixA = readMatrixRaw(argv[2]);
		rawMatrixB = readMatrixRaw(argv[3]);
		std::cout << "Input matrixes read from: " << argv[2] << ", " << argv[3] << std::endl;
	}
	else {
		rawMatrixA = {1, 1, 3};
		rawMatrixB = {1, 1, 4};
		std::cout << "Standart input matrixes taken" << std::endl;
	}
	
	if (0) {
		// 597ms
		// int N = 1000;
		// 398 ms
		// int N = 500;
		int N = 0;
		rawMatrixA.resize((size_t)2 + N * N);
		rawMatrixA[0] = rawMatrixA[1] = N;
		rawMatrixB = rawMatrixA;
	}
	
	resultN = rawMatrixA[0];
	resultM = rawMatrixB[1];
	rawMatrixBTransposed = transposeMatrixRaw(rawMatrixB);
	
	rawResult.resize((size_t)2 + resultN * resultM);
	rawResult[0] = resultN;
	rawResult[1] = resultM;
	
	std::string outputName = (argc >= 5) ? argv[4] : "matrix.out";
	
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
		
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();
		
		cl::Context context(device);
		
		cl::CommandQueue queue(context, device);
		
		cl::Program::Sources kernelSources{kernelMultiplySource};
		
		cl::Program program(context, kernelSources);
		program.build({device});
		
		cl::Kernel kernel(program, kernelMultiplyName.c_str());
		
		/*
		std::vector<cl_int> memoryData = concatenate(concatenate(rawMatrixA, rawMatrixBTransposed), rawResult);
		size_t memoryDataSizeInBytes = memoryData.size() * sizeof(cl_int);
		cl::Buffer buffer(context, CL_MEM_READ_WRITE, memoryDataSizeInBytes);
		queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, memoryDataSizeInBytes, memoryData.data());
		 */
		
		cl::Buffer bufferA(context, CL_MEM_READ_ONLY, rawMatrixA.size() * sizeof(cl_int));
		queue.enqueueWriteBuffer(bufferA, CL_FALSE, 0, rawMatrixA.size() * sizeof(cl_int), rawMatrixA.data());
		cl::Buffer bufferB(context, CL_MEM_READ_ONLY, rawMatrixBTransposed.size() * sizeof(cl_int));
		queue.enqueueWriteBuffer(bufferB, CL_FALSE, 0, rawMatrixBTransposed.size() * sizeof(cl_int),
		                        rawMatrixBTransposed.data());
		
		cl::Buffer bufferResult(context, CL_MEM_READ_WRITE, rawResult.size() * sizeof(cl_int));
		//queue.enqueueWriteBuffer(bufferResult, CL_TRUE, 0, rawResult.size() * sizeof(cl_int), rawResult.data());
		
		kernel.setArg(0, bufferA);
		kernel.setArg(1, bufferB);
		kernel.setArg(2, bufferResult);
		
		cl::Event event;
		queue.enqueueNDRangeKernel(kernel, cl::NullRange,
		                           cl::NDRange((size_t)resultN, (size_t)resultM),
		                           cl::NullRange, nullptr, &event);
		queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, rawResult.size() * sizeof(cl_int), rawResult.data());
		
		
		event.wait();
		end = std::chrono::system_clock::now();
		
		long int elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>
				(end-start).count();
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);
		
		std::cout << "finished computation at " << std::ctime(&end_time)
		          << "elapsed time: " << elapsed_seconds << "s\n";
		
		writeMatrixRaw(outputName, rawResult);
	}
	catch (cl::Error &error) {
		std::cerr << "ERROR: " << error.what() << "(" << error.err() << ")" << std::endl;
		return EXIT_FAILURE;
	}
	
	return EXIT_SUCCESS;
}