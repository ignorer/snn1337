cmake_minimum_required(VERSION 3.1)
project(GPU)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL)

add_subdirectory(FullyConnectedSNN)
add_subdirectory(Perceptron)