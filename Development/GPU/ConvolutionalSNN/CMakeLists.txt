project(ConvolutionalSNN)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${GPU_SOURCE_DIR}/include)

set(SOURCE_FILES Network.cpp Layer.cpp ConvLayer.cpp DenseLayer.cpp PoolLayer.cpp InputLayer.cpp NetworkInitializer.cpp InputReader.cpp main.cpp)
set(HEADER_FILES Network.h Layer.h ConvLayer.h DenseLayer.h PoolLayer.h InputLayer.h NetworkInitializer.h InputReader.h)
add_executable(ConvolutionalSNN ${SOURCE_FILES})

target_link_libraries(ConvolutionalSNN ${OpenCL_LIBRARY})
set_property(TARGET ConvolutionalSNN PROPERTY CXX_STANDARD 14)

# copy resources to working directory
file(COPY res/network DESTINATION ./res)
file(COPY mnist DESTINATION .)