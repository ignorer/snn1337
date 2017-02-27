set(MODULE_NAME Cheshear)

project(${MODULE_NAME})

set(CMAKE_CXX_STANDARD 14)

# set include directories
set(CUDA_ROOT "/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0")

include_directories(${OPENCL_INCLUDE_DIRS})
include_directories(${CUDA_ROOT}/include/CL)
include_directories(${CUDA_ROOT}/include)
include_directories(${task1_SOURCE_DIR}/include)

# set source files
set(SOURCE_FILES main.cpp)
add_executable(${MODULE_NAME} ${SOURCE_FILES})

# linking
target_link_libraries(${MODULE_NAME} ${OpenCL_LIBRARY})

# copy kernel file to bin directory
#file(COPY myFirstKernel.cl DESTINATION .)
