import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

mod = SourceModule("""
__global__ void sum2arrays(float *a, float *b, float *c,int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx<N) c[idx] = a[idx] + b[idx];
}
""")

if __name__ == "__main__":

    N = 100

    a = np.linspace(0,N-1,N).astype(np.float32)
    b = np.linspace(0,N-1,N).astype(np.float32)
    c = np.zeros_like(a)

    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)

    a = a*a
    b = b*(b-1)

    func = mod.get_function("sum2arrays")
    func(a_gpu, b_gpu, c_gpu, np.uint32(N), block=(16,1,1), grid=(int((N-1)/4+1),1,1))

    cuda.memcpy_dtoh(c, c_gpu)

    for i in range(N):
        print (a[i]," + ", b[i], " = ", c[i])
