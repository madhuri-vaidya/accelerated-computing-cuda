// Run Command
// !nvcc -o vector-add 07-vector-add/01-vector-add.cu -run

#include <stdio.h>

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; i++) 
  { 
    a[i] = num;
  }
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int i;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  while(i < N) {
    result[i] = a[i] + b[i];
    i += (blockDim.x * gridDim.x);
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  //a = (float *)malloc(size);
  //b = (float *)malloc(size);
  //c = (float *)malloc(size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  addVectorsInto<<<32, 256>>>(c, a, b, N);
  cudaDeviceSynchronize();

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
