
#include <stdio.h>
#include <cuda.h>

__global__ void sum2arrays(float *a, float *b, float *c,int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx<N) c[idx] = a[idx] + b[idx];
}

int main(void)
{
  float *d_a, *d_b, *d_c;
  float *h_a, *h_b, *h_c;

  const int N = 100;

  size_t size = N * sizeof(float);

  h_a = (float *)malloc(size);
  h_b = (float *)malloc(size);
  h_c = (float *)malloc(size);
  cudaMalloc((void **) &d_a, size);
  cudaMalloc((void **) &d_b, size);
  cudaMalloc((void **) &d_c, size);

  for (int i=0; i<N; i++) h_a[i] = float(i)*float(i);
  for (int i=0; i<N; i++) h_b[i] = float(i)*(float(i)-1);

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  dim3 DimGrid((N-1)/16 + 1, 1, 1);
  dim3 DimBlock(16, 1, 1);
  sum2arrays <<< DimGrid, DimBlock >>> (d_a, d_b, d_c, N);

  cudaMemcpy(h_c, d_c, sizeof(float)*N, cudaMemcpyDeviceToHost);

  for (int i=0; i<N; i++) printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);

  free(h_a); cudaFree(d_a);
  free(h_b); cudaFree(d_b);
  free(h_c); cudaFree(d_c);


}
