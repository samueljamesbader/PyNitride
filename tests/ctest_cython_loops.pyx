cimport cython
cimport numpy as cnp
import numpy as np
from libc.math cimport pow, exp
from functools import wraps
cnp.import_array()

from pynitride.util.cython_loops cimport dimsimple

cdef double timesz(double x, void* cargs):
    cdef double z
    cdef double* args
    args=<double*>cargs
    z = args[0]
    return z*x

def ctest_dimsimple():
    sam=np.linspace(2,5.0)
    lex=np.empty_like(sam)

    cdef double args[1]
    cdef void* cargs

    args=[3]
    lex2=dimsimple(sam,timesz,args=args)

    print(lex2)
    #tryme()
