//
// Created by V on 26.02.2017.
//

//#include <cl.hpp>
#include <CL/cl.hpp>
#include <iostream>

int main(){
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
    cl::Context context(default_device);
    cl::Program::Sources sources;
    std::string kernel_code=
                        "   void kernel f(global int* A){       "
                                "       A[get_global_id(0)]=get_global_id(0);                 "
                                "   }";
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    cl::Program program(context,sources);
    if(program.build({default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(int)*10);

    int A[10];

    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,default_device);

    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(int)*10,A);

    //alternative way to run the kernel
    cl::Kernel kernel_f=cl::Kernel(program,"f");
    kernel_f.setArg(0,buffer_A);
    queue.enqueueNDRangeKernel(kernel_f,cl::NullRange,cl::NDRange(10),cl::NullRange);
    queue.finish();

    queue.enqueueReadBuffer(buffer_A,CL_TRUE,0,sizeof(int)*10,A);

    std::cout<<" result: \n";
    for(int i=0;i<10;i++){
        std::cout<<A[i]<<" ";
    }

    return 0;
}

