#include <cstdlib>
#include <iostream>
#include <fstream>

#include <cl.hpp>

// Matrix multiplication function called by MatMulKernel()

using namespace std;

void RandMatrix(int *Matr, int rows, int columns) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            Matr[j + i * columns] = 1 + rand() % 50;
        }
    }
}

void printMatrix(int *matr, int rows, int columns) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            cout << matr[j + i * columns] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void transpose(int *matr, int rows, int columns) {
    int *Transpose = new int[columns * rows];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            Transpose[j + i * columns] = matr[i + j * rows];
        }
    }
    for (int i = 0; i < columns * rows; ++i) {
        matr[i] = Transpose[i];
    }
    delete[](Transpose);
}

int main() {
    int Size;
    cin >> Size;
    int A;
    int *A_m = new int[Size * Size];
    int *B_m = new int[Size * Size];
    int *C_m = new int[Size * Size];
    RandMatrix(A_m, Size, Size);
    RandMatrix(B_m, Size, Size);
    // printMatrix(B_m, Size, Size);
    transpose(B_m, Size, Size);
    // printMatrix(B_m, Size, Size);

    std::vector <cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    //get default device of the default platform
    std::vector <cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device = all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
    cl::Context context(default_device);
    std::ifstream sourceFile("OpenCLFile1.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));;
    cl::Program program = cl::Program(context, source);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        getchar();
        exit(1);
    }
    size_t Msize = Size * Size * sizeof(int);
    cl::Buffer aElements(context, CL_MEM_READ_WRITE, Msize);
    cl::Buffer bElements(context, CL_MEM_READ_WRITE, Msize);
    cl::Buffer cElements(context, CL_MEM_READ_WRITE, Msize);
    cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(int));

    cl::CommandQueue queue(context, default_device);

    queue.enqueueWriteBuffer(A_elements, CL_TRUE, 0, Msize, &A_m[0]);
    queue.enqueueWriteBuffer(B_elements, CL_TRUE, 0, Msize, &B_m[0]);

    cl::Kernel kernel(program, "MatrixMult");
    int iArg = 0;
    cl_uint i = 0;
    kernel.setArg(i++, A_elements);
    kernel.setArg(i++, B_elements);
    kernel.setArg(i++, C_elements);
    kernel.setArg(i++, buffer_A);
    cl::NDRange global(Size * Size);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
    queue.finish();

    //read result C from the device to array C
    queue.enqueueReadBuffer(C_elements, CL_TRUE, 0, Msize, C_m);
    cout << "A_m" << endl;
    printMatrix(A_m, Size, Size);
    cout << "B_m" << endl;
    printMatrix(B_m, Size, Size);
    cout << "C_m" << endl;
    printMatrix(C_m, Size, Size);
    return 0;
}
