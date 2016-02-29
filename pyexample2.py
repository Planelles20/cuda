import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

mod = SourceModule("""
__global__ void Multy2Matrix(int m, int n, int k, float* A, float* B, float* C)
{
  int Row = blockIdx.y*blockDim.y+threadIdx.y;
  int Col = blockIdx.x*blockDim.x+threadIdx.x;
  if ((Row < n) && (Col < k)) {
    float Cvalue = 0.0;
    for (int i = 0; i < m; ++i)  Cvalue += A[Row*m+i] * B[Col+i*k];
    C[Row*k+Col] = Cvalue;
  }
}
""")

#  B(KxM) x A(MxN) = C(KxN)

if __name__ == "__main__":

    N = 9
    M = 5
    K = 3

    a = np.zeros(N*M).astype(np.float32)
    b = np.zeros(M*K).astype(np.float32)
    c = np.zeros(N*K).astype(np.float32)

    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)

    for j in range(N):
        for i in range(M):
            a[j*M+i] = j+i

    for j in range(M):
        for i in range(K):
            if(i<j):
                r = j
            else:
                r = i
            b[j*K+i] = r

    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    func = mod.get_function("Multy2Matrix")
    threads = max(N,K) 
    func(np.uint32(M),  np.uint32(N),  np.uint32(K), a_gpu, b_gpu, c_gpu,
        block=(threads,threads,1), grid=(int((N-1)/threads+1),int((K-1)/threads+1),1))

    cuda.memcpy_dtoh(c, c_gpu)

    A = np.zeros((M,N))
    B = np.zeros((K,M))
    C = np.zeros((K,N))

    for j in range(N):
        for i in range(M):
            A[i,j] = a[j*M+i]

    for j in range(M):
        for i in range(K):
            B[i,j] = b[j*K+i]

    for j in range(N):
        for i in range(K):
            C[i,j] = c[j*K+i]

    C2 = np.dot(B,A)

    print("B(KxM) x A(MxN) = C(KxN)")
    print("\nMatriz A:\n", A)
    print("\nMatriz B:\n", B)
    print("\nMatriz C:\n", C)
    print("\nMatriz C2 (numpay.dot(B,A) = C2):\n", C2)
