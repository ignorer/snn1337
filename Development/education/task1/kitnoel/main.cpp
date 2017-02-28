#include <iostream>
#include <fstream>
#include <strstream>
#include <vector>
#include <memory>
#include <iterator>

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

std::string loadSourceCode(std::string filename) {
	std::ifstream sourceFile;
	sourceFile.open(filename);
	if (!sourceFile.is_open()) {
		std::cout << "Failed to load source file" << std::endl;
		exit(1);
	}
	return std::string((std::istreambuf_iterator<char>(sourceFile)), std::istreambuf_iterator<char>());
}

const std::string sourceCode = loadSourceCode("myFirstKernel.cl");

int main() {
	unsigned int N = 10;

	try {
		std::vector<cl::Platform> all_platforms;
		cl::Platform::get(&all_platforms);
		if (all_platforms.size() == 0) {
			std::cout << " No platforms found. Check OpenCL installation!\n";
			exit(1);
		}
		cl::Platform platform = all_platforms[0];
		std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

		std::vector<cl::Device> all_devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
		if (all_devices.size() == 0) {
			std::cout << " No devices found. Check OpenCL installation!\n";
			exit(1);
		}
		cl::Device device = all_devices[0];
		std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

		cl::Context context(device);

		cl::CommandQueue queue(context, device);

		cl::Program::Sources sources{{sourceCode.c_str(), sourceCode.size()}};

		cl::Program program(context, sources);

		try {
			program.build({device});
		}
		catch (cl::Error error) {
			std::string build_log;
			program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &build_log);
			std::cout << build_log << std::endl;
		}

		std::vector<cl_int> arr(N);
		cl::Buffer buffer(context, CL_MEM_READ_WRITE, arr.size() * sizeof(cl_int));

		// create kernel functor
		auto kernel = cl::make_kernel<cl::Buffer&>(program, "job");
		kernel(cl::EnqueueArgs(queue, cl::NDRange(N)), buffer).wait();

		cl::copy(queue, buffer, arr.begin(), arr.end());

		std::cout << "Result: \n";
		for (int i = 0; i < N; i++) {
			std::cout << arr[i] << " ";
		}

	}
	catch (cl::Error& error) {
		std::cerr << "ERROR: " << error.what() << "(" << error.err() << ")" << std::endl;
		return EXIT_FAILURE;
	}
	return 0;
}