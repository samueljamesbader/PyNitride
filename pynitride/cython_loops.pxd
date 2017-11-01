cimport cython
cimport numpy as cnp
from cython cimport numeric

cdef cnp.ndarray dimsimple(cnp.ndarray inarr, double (*func)(double,void*),void* args=*,cnp.ndarray outarr=*)
r""" Does Sphinx find this in a pxd?

:param inarr:
:param func:
:param args:
:param outarr:
:return:
"""


cdef cnp.ndarray gridinput(cnp.ndarray inarr1, cnp.ndarray inarr2,
                           double (*func)(double,double,void*),void* args=*,
                           cnp.ndarray outarr=*)

