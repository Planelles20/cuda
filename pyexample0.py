import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void square_array(float *a,float *b, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx<N) b[idx] = a[idx] * a[idx];
}
""")

square_array = mod.get_function("square_array")

N = 10

a = np.linspace(0,N-1,N).astype(np.float32)
b = np.zeros_like(a)

square_array(
        drv.In(a), drv.Out(b), np.uint32(N),
        block=(int((N-1)/4+1),1,1), grid=(4,1,1))


for i in range(N):
    print (a[i], b[i])
