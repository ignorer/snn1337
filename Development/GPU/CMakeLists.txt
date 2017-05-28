cmake_minimum_required(VERSION 3.1)
project(GPU)

find_package(OpenCL)

add_subdirectory(Perceptron)
add_subdirectory(FullyConnectedSNN)
add_subdirectory(ConvolutionalSNN)

