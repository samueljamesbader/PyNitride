cimport cython
cimport numpy as cnp
import numpy as np
from libc.math cimport pow, exp
from functools import wraps

# Wong et al Solid-State Electronics Vol. 37, No. I, pp. 61~64, 1994
# http://dx.doi.org.proxy.library.cornell.edu/10.1016/0038-1101(94)90105-8
DEF ORDER=7
cdef double[ORDER] a, b1, b2, c
a[:]=[1,.353568,.192439,.122973,.077134,.036228,.008346]
b1[:]=[.76514793,.60488667,.19003355,2.00193968e-2,-4.12643816e-3,-4.70958992e-4,1.50071469e-4]
b2[:]=[.78095732,.57254453,.21419339,1.38432741e-2,-5.54949386e-3,6.48814900e-4,-2.84050520e-5]
c[:]=[.752253,.928195,.680839,25.7829,-553.636,3531.43,-3254.65]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray fd12_c(double[::1] x):
            
    cdef:
        cnp.ndarray outarr=np.empty(x.shape[0])
        double[::1] out=outarr
        int i,j,s
        double partial_sum=0
    
    for i in range(x.shape[0]):
        partial_sum=0
        if x[i]<=0:
            s=1
            for j in range(ORDER):
                partial_sum+=s*a[j]*exp((j+1)*x[i])
                s*=-1
        elif x[i]<=2:
            partial_sum=b1[ORDER-1]
            for j in range(1,ORDER):
                partial_sum=partial_sum*x[i] + b1[ORDER-j-1]
        elif x[i]<=5:
            partial_sum=b2[ORDER-1]
            for j in range(1,ORDER):
                partial_sum=partial_sum*x[i] + b2[ORDER-j-1]
        else:
            for j in range(ORDER):
                partial_sum+=c[j]*pow(x[i],1.5-2*j)
        out[i]=partial_sum
    return outarr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray fd12p_c(double[::1] x):
    cdef:
        cnp.ndarray outarr=np.empty(x.shape[0])
        double[::1] out=outarr
        int i,j,s
        double partial_sum=0
    
    for i in range(x.shape[0]):
        partial_sum=0
        if x[i]<=0:
            s=1
            for j in range(ORDER):
                partial_sum+=s*a[j]*(j+1)*exp((j+1)*x[i])
                s*=-1
        elif x[i]<=2:
            partial_sum=(ORDER-1)*b1[ORDER-1]
            for j in range(1,ORDER-1):
                partial_sum=partial_sum*x[i] + (ORDER-j-1)*b1[ORDER-j-1]
        elif x[i]<=5:
            partial_sum=(ORDER-1)*b2[ORDER-1]
            for j in range(1,ORDER-1):
                partial_sum=partial_sum*x[i] + (ORDER-j-1)*b2[ORDER-j-1]
        else:
            for j in range(ORDER):
                partial_sum+=(1.5-2*j)*c[j]*pow(x[i],.5-2*j)
        out[i]=partial_sum
    return outarr


#cdef numpywrapper(cnp.ndarray (*func)(double[::1]),x):
#    if hasattr(x,'__iter__'):
#        return func(np.asarray(x,dtype='float'))
#    else:
#        return func(np.asarray([x],dtype='float'))[0]
#
#def fd12(x):
#    return numpywrapper(&fd12_c,x)
#def fd12p(x):
#    return numpywrapper(&fd12p_c,x)
    
    
    
cdef numpywrapper(func):
    @wraps(func)
    def func2(x):
        if hasattr(x,'__iter__'):
            return func(np.asarray(x,dtype='float'))
        else:
            return func(np.asarray([x],dtype='float'))[0]
    return func2

@numpywrapper
def fd12(x): return fd12_c(x)

@numpywrapper    
def fd12p(x): return fd12p_c(x)