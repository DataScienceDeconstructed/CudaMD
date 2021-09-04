#include "Kernels.cuh"

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}


__global__
void calc_v2(float3* v, float* rValues, int threadMax){
  int index = threadMax*blockIdx.x + threadIdx.x;
  rValues[index] = v[index].x * v[index].x + v[index].y * v[index].y + v[index].z * v[index].z ;
}
