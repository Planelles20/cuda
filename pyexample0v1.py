import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

N = 10

#But wait–a consists of double precision numbers, but most nVidia devices only support single precision:
a = np.linspace(0,N-1,N).astype(np.float32)
#Finally, we need somewhere to transfer data to, so we need to allocate memory on the device:
a_gpu = cuda.mem_alloc(a.nbytes)
#As a last step, we need to transfer the data to the GPU:
cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
__global__ void square_array(float *a, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx<N) a[idx] = a[idx] * a[idx];
}
""")

func = mod.get_function("square_array")
func(a_gpu, np.uint32(N), block=(int((N-1)/4+1),1,1), grid=(4,1,1))

a_square = np.empty_like(a)
cuda.memcpy_dtoh(a_square, a_gpu)

for i in range(N):
    print (a[i], a_square[i])
