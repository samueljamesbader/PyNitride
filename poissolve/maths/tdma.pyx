cimport cython
cimport numpy as cnp
import numpy as np

# https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
# but with indices shifted

# assumes a[0]=0, c[n-1]=0
@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray tdma_c(double[::1] a, double[::1] b, double[::1] c, double[::1] d):
    cdef:
        int N=b.shape[0]
        int k
        double m
        cnp.ndarray x_arr=np.empty(N)
        double[::1] x=x_arr
    for k in range(1,N):
        m=a[k]/b[k-1]
        b[k]-=m*c[k-1]
        d[k]-=m*d[k-1]
    for k in range(N):
        x[k]=d[k]/b[k]
    for k in range(N-2,-1,-1):
        x[k]=(d[k]-c[k]*x[k+1])/b[k]
    return x_arr

def tdma(a,b,c,d,copy=True):
    if copy:
        a=a.copy()
        b=b.copy()
        c=c.copy()
        d=d.copy()
    return tdma_c(a,b,c,d)
