// Run command
// nvcc -o hello-gpu 01-hello/01-hello-gpu.cu -run

#include <iostream>
#include <chrono>
#include <ctime>
#include <stdio.h>


using namespace std;

void helloCPU() { cout << "Hello from the CPU" << endl; }

/*
* Refactor the `helloGPU` definition to be a kernel that can be launched on the GPU. 
* Update its message to read "Hello from the GPU!"
*/

__global__ void helloGPU() { printf("Hello from the GPU\n"); }


int main()
{
    /*
    * Refactor this call to `helloGPU` so that it launches as a kernel on the GPU.
    */
    auto start = chrono::system_clock::now();
    helloGPU<<<1, 1>>>();

    /*
    * Add code below to synchronize on the completion of the `helloGPU` kernel completion before continuing the CPU thread.
    */

    cudaDeviceSynchronize();

    auto gpu1 = chrono::system_clock::now();
    chrono::duration<double> gpu_duration_1 = gpu1 - start;
    cout << "Time taken for GPU function is " << gpu_duration_1.count() << endl;

    helloCPU();

    auto cpu = chrono::system_clock::now();
    chrono::duration<double> cpu_duration = cpu - gpu1;
    cout << "Time taken for CPU function is " << cpu_duration.count() << endl;

    helloGPU<<<1, 1>>>();
    cudaDeviceSynchronize();

    auto gpu2 = chrono::system_clock::now();
    chrono::duration<double> gpu_duration_2 = gpu2 - cpu;
    cout << "Time taken for GPU function is " << gpu_duration_2.count() << endl;

    return 0;
}
