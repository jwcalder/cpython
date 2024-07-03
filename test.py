import numpy as np
import cmodules.synchronization as sync

m = 20
n = 3
p = 4

u = np.random.rand(m,n,p)
v = np.random.rand(m,n,p)
w = np.zeros((m,n,p))

#It is VERY important to form as a contiguous array in memory, as below, for passing to C code
u = np.ascontiguousarray(u,dtype=np.float64)
v = np.ascontiguousarray(v,dtype=np.float64)
w = np.ascontiguousarray(w,dtype=np.float64)

sync.add_arrays(u,v,w)
print(u + v - w)

sync.subtract_arrays(u,v,w)
print(u - v - w)

sync.multiply_arrays(u,v,w)
print(u * v - w)

sync.divide_arrays(u,v,w)
print(u / v - w)

#The code below computes w[i,:,:] = u[i,:,:].T@v[i,:,:] for all i
w = np.zeros((m,p,p))
w = np.ascontiguousarray(w,dtype=np.float64)
sync.transpose_matmult_arrays(u,v,w)
for i in range(m):
    print(w[i,:,:]-u[i,:,:].T@v[i,:,:])

