#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>

#include <cl.hpp>

// Matrix multiplication function called by MatMulKernel()

using namespace std;

void generateRandomMatrix(vector<int>& matrix, int rows, int columns) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            matrix[j + i * columns] = 1 + rand() % 50;
        }
    }
}

void printMatrix(vector<int>& matrix, int rows, int columns) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            cout << matrix[j + i * columns] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void transpose(vector<int>& matrix, unsigned int rows, unsigned int columns) {
    vector<int> temp(columns * rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            temp[j + i * columns] = matrix[i + j * rows];
        }
    }
    matrix = temp;
}

int main() {
    unsigned int size = 5;

    vector<int> matrixA(size * size);
    vector<int> matrixB(size * size);
    vector<int> matrixC(size * size);
    generateRandomMatrix(matrixA, size, size);
    generateRandomMatrix(matrixB, size, size);
    // printMatrix(B_m, size, size);
    transpose(matrixB, size, size);
    // printMatrix(B_m, size, size);

    std::vector<cl::Platform> allPlatforms;
    cl::Platform::get(&allPlatforms);
    if (allPlatforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

    // get default device and context
    cl::Platform defaultPlatform = allPlatforms[0];
    std::vector<cl::Device> allDevices;
    defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
    if (allDevices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device defaultDevice = allDevices[0];
    cl::Context context(defaultDevice);

    // load kernel
    std::ifstream sourceFile("OpenCLFile1.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));;
    cl::Program program = cl::Program(context, source);
    if (program.build({defaultDevice}) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice) << "\n";
        getchar();
        exit(1);
    }
    size_t mSize = size * size * sizeof(int);
    cl::Buffer aElements(context, CL_MEM_READ_WRITE, mSize);
    cl::Buffer bElements(context, CL_MEM_READ_WRITE, mSize);
    cl::Buffer cElements(context, CL_MEM_READ_WRITE, mSize);

    cl::CommandQueue queue(context, defaultDevice);

    queue.enqueueWriteBuffer(aElements, CL_TRUE, 0, mSize, matrixA.data());
    queue.enqueueWriteBuffer(bElements, CL_TRUE, 0, mSize, matrixB.data());

    cl::Kernel kernel(program, "MatrixMult");
    cl_uint i = 0;
    kernel.setArg(i++, aElements);
    kernel.setArg(i++, bElements);
    kernel.setArg(i++, cElements);
    kernel.setArg(i++, size);

    cl::NDRange global(size);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
    queue.finish();

    //read result C from the device to array C
    // cout << "C_m" << endl;
    //   printMatrix(matrixC, size, size);
    queue.enqueueReadBuffer(cElements, CL_TRUE, 0, mSize, matrixC.data());
//    cout << "A_m" << endl;
    printMatrix(matrixA, size, size);
//    cout << "B_m" << endl;
    printMatrix(matrixB, size, size);
//    cout << "C_m" << endl;
    printMatrix(matrixC, size, size);
    return 0;
}