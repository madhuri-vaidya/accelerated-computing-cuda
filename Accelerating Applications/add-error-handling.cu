// Run Command
// !nvcc -o add-error-handling 06-errors/01-add-error-handling.cu -run

#include <stdio.h>

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

__global__ void doubleElements(int *a, int N)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N + stride; i += stride)
  {
    a[i] *= 2;
  }
}

bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (a[i] != i*2) return false;
  }
  return true;
}

int main()
{
  /*
   * Add error handling to this source code to learn what errors
   * exist, and then correct them. Googling error messages may be
   * of service if actions for resolving them are not clear to you.
   */

  int N = 10000;
  int *a;
  cudaError_t err_malloc, err_last, err_sync;

  size_t size = N * sizeof(int);
  err_malloc = cudaMallocManaged(&a, size);
  if (err_malloc != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(err_malloc)); }
 
  init(a, N);

  size_t threads_per_block = 1024;
  size_t number_of_blocks = 32;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  err_last = cudaGetLastError();
  if (err_last != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(err_last)); }
  
  
  err_sync = cudaDeviceSynchronize();
  if (err_sync != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(err_sync)); }

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(a);
}
