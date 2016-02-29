#include <stdio.h>
#include <cuda.h>
#include <iostream>

__global__ void Multy2Matrix(int m, int n, int k, float* A, float* B, float* C)
{
  int Row = blockIdx.y*blockDim.y+threadIdx.y;
  int Col = blockIdx.x*blockDim.x+threadIdx.x;
  if ((Row < n) && (Col < k)) {
    float Cvalue = 0.0;
    for (int i = 0; i < m; ++i)  Cvalue += A[Row*n+i] * B[Col+i*m];
    C[Row+Col*k] = Cvalue;
  }
}

/*
  A(NxM) x B(MxK) = C(NxK)
*/

int main(void)
{
  float *d_a, *d_b, *d_c;
  float *h_a, *h_b, *h_c;

  const int N = 2;
  const int M = 2;
  const int K = 2;

  size_t size_a = N * M * sizeof(float);
  size_t size_b = M * K * sizeof(float);
  size_t size_c = N * K * sizeof(float);

  h_a = (float *)malloc(size_a);
  h_b = (float *)malloc(size_b);
  h_c = (float *)malloc(size_c);
  cudaMalloc((void **) &d_a, size_a);
  cudaMalloc((void **) &d_b, size_b);
  cudaMalloc((void **) &d_c, size_c);

  for (int i=0; i<N; i++){
    for (int j=0; j<M; j++){
      h_a[j*N+i] = i+j;
    }
  }

  for (int i=0; i<M; i++){
    for (int j=0; j<K; j++){
      h_b[j*M+i] = (i<j ? j:i);
    }
  }

  cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

  const int threads = 256;

  dim3 DimGrid((N-1)/threads + 1, (K-1)/threads + 1, 1);
  dim3 DimBlock(threads, threads, 1);
  Multy2Matrix <<< DimGrid, DimBlock >>> (M, N, K, d_a, d_b, d_c);

  cudaMemcpy(h_c, d_c, sizeof(float)*N*K, cudaMemcpyDeviceToHost);

  std::cout << "\nA matrix \n";
  for (int i=0; i<N; i++){
    for (int j=0; j<M; j++){
      std::cout << h_a[j*N+i] << " ";
    }
    std::cout << "\n";
  }

  std::cout << "\nB matrix \n";
  for (int i=0; i<M; i++){
    for (int j=0; j<K; j++){
      std::cout << h_b[j*M+i] << " ";
    }
    std::cout << "\n";
  }

  std::cout << "\nC matrix \n";
  for (int i=0; i<N; i++){
    for (int j=0; j<K; j++){
      std::cout << h_c[j*N+i] << " ";
    }
    std::cout << "\n";
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
