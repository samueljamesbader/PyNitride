r""" Provides several cython-optimized mathematical utilities for band diagram generation.

These functions are validated and speed-checked by the :py:mod:`tests.test_maths` module.
"""

cimport cython
cimport numpy as cnp
import numpy as np
from libc.math cimport pow, exp
from functools import wraps
cnp.import_array()

##########
### Tridiagonal matrix algorithm
##########

# c-function to implement tdma. See :py:func:`~pynitride.poissolve.maths.tmda` for more.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray tdma_c(double[::1] a, double[::1] b, double[::1] c, double[::1] d):
    cdef:
        int N=b.shape[0]
        int k
        double m
        cnp.ndarray x_arr=np.empty(N)
        double[::1] x=x_arr

    # Implementation precisely follows
    # https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    # Except that these arrays are zero-indexed!
    # Makes the same assumption: a[0]=0, c[n-1]=0
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
    r"""Implements the tridiagonal matrix algorithm.

    Solves :math:`Mx=d` where :math:`M` is a tridiagonal matrix described by three diagonals :math:`a,b,c`.
    A good discussion of the algorithm is available at
    `CFD-Online <https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)>`_.  All
    arguments are given as 1-D numpy arrays of the same length.

    :param a: the lower subdiagonal of :math:`M`. This should include a single leading zero so it is the same length as *b*.
    :param b: the diagonal of :math:`M`.
    :param c: the upper subdiagonal of :math:`M`. This should include a single trailing zero so it is the same length as *b*.
    :param d: the desired result of :math:`Mx`.  Should be the same length as :math:`b`.
    :param copy: If true, arrays will be copied (default); otherwise they will be overwritten.

    :return: :math:`x`, the solution vector as a numpy array.
    """

    N=len(b)
    assert len(a)==N, 'a should be the same length as b.'
    assert len(c)==N, 'c should be the same length as b.'
    assert len(d)==N, 'd should be the same length as b.'
    if copy:
        a=a.copy()
        b=b.copy()
        c=c.copy()
        d=d.copy()
    return tdma_c(a,b,c,d)

##########
### Fermi-Dirac integrals
##########


# These values come from Table 6 for use in the Fermi-Dirac 1/2 integrals
# Wong et al Solid-State Electronics Vol. 37, No. I, pp. 61~64, 1994
# http://dx.doi.org/10.1016/0038-1101(94)90105-8
DEF ORDER=7
cdef double[ORDER] a, b1, b2, c, ap, b1p, b2p, cp
a[:]=[1,.353568,.192439,.122973,.077134,.036228,.008346]
b1[:]=[.76514793,.60488667,.19003355,2.00193968e-2,-4.12643816e-3,-4.70958992e-4,1.50071469e-4]
b2[:]=[.78095732,.57254453,.21419339,1.38432741e-2,-5.54949386e-3,6.48814900e-4,-2.84050520e-5]
c[:]=[.752253,.928195,.680839,25.7829,-553.636,3531.43,-3254.65]

# These values will be used by the Fermi-Dirac 1/2 Derivative function
# They come from modifying the above values by differentiating each expression in piecewise
# from Equations 24-26 in the same reference.
for j in range(ORDER):
    ap[j] =a[j]*(j+1)
    b1p[ORDER-j-1]=(ORDER-j-1)*b1[ORDER-j-1]
    b2p[ORDER-j-1]=(ORDER-j-1)*b2[ORDER-j-1]
    cp[j] =(1.5-2*j)*c[j]
############################################
@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray fd12_2d_c(double[:,::] x):

    cdef:
        cnp.ndarray outarr=np.empty((x.shape[0],x.shape[1]))
        double[:,::] out=outarr
        int ic,i,j,s
        double partial_sum=0

    for ic in range(x.shape[0]):
        for i in range(x.shape[1]):
            partial_sum=0
            if x[ic,i]<=-10:
                partial_sum=exp(x[ic,i])
            elif x[ic,i]<=0:
                s=1
                for j in range(ORDER):
                    partial_sum+=s*a[j]*exp((j+1)*x[ic,i])
                    s*=-1
            elif x[ic,i]<=2:
                partial_sum=b1[ORDER-1]
                for j in range(1,ORDER):
                    partial_sum=partial_sum*x[ic,i] + b1[ORDER-j-1]
            elif x[ic,i]<=5:
                partial_sum=b2[ORDER-1]
                for j in range(1,ORDER):
                    partial_sum=partial_sum*x[ic,i] + b2[ORDER-j-1]
            else:
                for j in range(ORDER):
                    partial_sum+=c[j]*pow(x[ic,i],1.5-2*j)
            out[ic,i]=partial_sum
    return outarr

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray fd12p_2d_c(double[:,::] x):
    cdef:
        cnp.ndarray outarr=np.empty((x.shape[0],x.shape[1]))
        double[:,::] out=outarr
        int ic,i,j,s
        double partial_sum=0

    for ic in range(x.shape[0]):
        for i in range(x.shape[1]):
            partial_sum=0
            if x[ic,i]<=-10:
                partial_sum=exp(x[ic,i])
            elif x[ic,i]<=0:
                s=1
                for j in range(ORDER):
                    partial_sum+=s*ap[j]*exp((j+1)*x[ic,i])
                    s*=-1
            elif x[ic,i]<=2:
                partial_sum=b1p[ORDER-1]
                for j in range(1,ORDER-1):
                    partial_sum=partial_sum*x[ic,i] + b1p[ORDER-j-1]
            elif x[ic,i]<=5:
                partial_sum=b2p[ORDER-1]
                for j in range(1,ORDER-1):
                    partial_sum=partial_sum*x[ic,i] + b2p[ORDER-j-1]
            else:
                for j in range(ORDER):
                    partial_sum+=cp[j]*pow(x[ic,i],.5-2*j)
            out[ic,i]=partial_sum
    return outarr
##################################################









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



def numpywrapper(func):
    @wraps(func)
    def func2(x):
        print(type(x))
        print(x.shape)
        if hasattr(x,'__iter__'):
            return np.reshape(func(np.atleast_2d(np.asarray(x,dtype='float'))),x.shape)
            #return func(np.asarray(x,dtype='float'))
        else:
            return func(np.asarray([[x]],dtype='float'))[0,0]
    return func2

@numpywrapper
def fd12(x):
    return fd12_2d_c(x)

@numpywrapper
def fd12p(x):
    #print(x.shape)
    return fd12p_2d_c(x)
