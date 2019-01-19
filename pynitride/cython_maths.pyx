r""" Cython-optimized mathematical utilities for the Poisson-Schrodinger problem.

This module contains a couple useful high-performance math functions, including
a tridiagonal matrix solver
and an approximate Fermi-Dirac integral of order 1/2 and -1/2.
These functions are compiled by Cython, and are tested for accuracy and speed by the
:py:mod:`~tests.test_maths` module.
"""

cimport cython
cimport numpy as cnp
import numpy as np
from libc.math cimport pow, exp, sqrt
from scipy.sparse import lil_matrix
from functools import wraps
cnp.import_array()
from pynitride.cython_loops cimport dimsimple
from cpython cimport bool

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


# c-function to implement Fermi-Dirac 1/2 integral.  See :py:func:`~pynitride.poissolve.maths.fd12` for more.
# args is disregarded but important to match the signature required by :py:func:`~pynitride.util.cython_loops.dimsimple`
cdef double fd12_scalar(double x,void* args):

    cdef:
        double partial_sum
        long j
        long s

    # Follows the sums from Wong et al (see documentation for fd12)
    partial_sum=0
    if x<=-10:
        partial_sum=exp(x)
    elif x<=0:
        s=1
        for j in range(ORDER):
            partial_sum+=s*a[j]*exp((j+1)*x)
            s*=-1
    elif x<=2:
        partial_sum=b1[ORDER-1]
        for j in range(1,ORDER):
            partial_sum=partial_sum*x + b1[ORDER-j-1]
    elif x<=5:
        partial_sum=b2[ORDER-1]
        for j in range(1,ORDER):
            partial_sum=partial_sum*x + b2[ORDER-j-1]
    else:
        for j in range(ORDER):
            partial_sum+=c[j]*pow(x,1.5-2*j)
    return partial_sum


# c-function to implement Fermi-Dirac 1/2 Integral Derivative.  See :py:func:`~pynitride.poissolve.maths.fd12p` for more.
# args is disregarded but important to match the signature required by :py:func:`~pynitride.util.cython_loops.dimsimple`
cdef double fd12p_scalar(double x,void* args):
    cdef:
        double partial_sum
        long j
        long s
    partial_sum=0

    # Code is the same as in fd12_scalar except using the primed coefficients,
    # and changing the power in the final sum
    if x<=-10:
        partial_sum=exp(x)
    elif x<=0:
        s=1
        for j in range(ORDER):
            partial_sum+=s*ap[j]*exp((j+1)*x)
            s*=-1
    elif x<=2:
        partial_sum=b1p[ORDER-1]
        for j in range(1,ORDER-1):
            partial_sum=partial_sum*x + b1p[ORDER-j-1]
    elif x<=5:
        partial_sum=b2p[ORDER-1]
        for j in range(1,ORDER-1):
            partial_sum=partial_sum*x + b2p[ORDER-j-1]
    else:
        for j in range(ORDER):
            partial_sum+=cp[j]*pow(x,.5-2*j)
    return partial_sum

# cdef cnp.ndarray dimsimple(double (*func)(double),x):
#     r"""Broadcasts the scalar function func over the numpy array x.
#
#     :param func: a cdef function which takes a double and returns a double.
#     :param x: a numpy float array of arbitrary shape and dimensionality
#     :return: an array of the same shape as ``x``, with the values obtained by calling ``func`` element-wise
#     """
#     x=np.asarray(x,dtype='float')
#     out=np.empty(x.shape,np.float)
#     cdef double xi
#
#     # This is an older syntax... should rewrite in terms of nditer when I get the chance
#     # but at this point that's just a difference of taste inside of a blackboxed function...
#     it=cnp.PyArray_MultiIterNew(2,<cnp.PyObject*>x,<cnp.PyObject*>out)
#     while cnp.PyArray_MultiIter_NOTDONE(it):
#         xi=(<double*>(cnp.PyArray_MultiIter_DATA(it,0)))[0]
#         (<double*>cnp.PyArray_MultiIter_DATA(it,1))[0]=func(xi)
#         cnp.PyArray_MultiIter_NEXT(it)
#     return out

# Table 6 and Equations 24-26
def fd12(x):
    r"""Implements the Fermi-Dirac integral of order 1/2 (Van Halen and Pulfrey approximation).

    This is essentially as given in Table 6 and Equations 24-26 of
    `(Wong et al Solid-State Electronics 1994) <http://dx.doi.org/10.1016/0038-1101(94)90105-8>`_, except that
    for very negative arguments (:math:`x < -10`), this function will replace the sum over integrals by simply
    the first term (:math:`e^x`).  Since, in a typical simulation, the Fermi-Dirac integral is often computed for
    arguments with very negative :math:`x` (ie Fermi-level in midgap), this often results in a several-times
    speedup.

    :param x: the argument to the Fermi-Dirac 1/2 integral, as a numpy array.
    :returns: the evaluation, as a numpy array.
    """
    #return dimsimple(fd12_scalar,x)
    return dimsimple(np.asarray(x,dtype=np.double),fd12_scalar)

def fd12p(x):
    r"""Implements the derivative of the :py:func:`~pynitride.poissolve.maths.fd12`.

    Computes :math:`\mathcal{F}_{1/2}'(x)`, which is equal to :math:`\mathcal{F}_{-1/2}(x)`.  The appropriate sums
    were obtained by simply differentiating the expressions used to form :py:func:`~pynitride.poissolve.maths.fd12`.

    :param x: the argument to the Fermi-Dirac -1/2 integral, as a numpy array.
    :returns: the evaluation, as a numpy array.
    """
    return dimsimple(np.asarray(x,dtype=np.double),fd12p_scalar)



# c-function to implement ionized dopant density
# args should be scalar double g
cdef double idd_scalar(double eta, void* args):
    cdef:
        double g
    g=(<double*> args)[0]
    if eta>500:
        return 0
    else:
        return 1/(1+g*exp(eta))

def idd(eta,g):
    cdef double g2
    g2=g
    return dimsimple(np.asarray(eta,dtype=np.double),idd_scalar,&g2)


# c-function to implement ionized dopant density derivative
# args should be scalar double g
cdef double iddd_scalar(double eta, void* args):
    cdef:
        double g
    g=(<double*> args)[0]
    if eta>500:
        return 0
    else:
        return g*exp(eta)/(1+g*exp(eta))**2

def iddd(eta,g):
    cdef:
        double g2
    g2=g
    return dimsimple(np.asarray(eta,dtype=np.double),iddd_scalar,&g2)


