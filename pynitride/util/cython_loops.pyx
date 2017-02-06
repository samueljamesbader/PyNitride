r""" Cython looooooops

:func dimsimple: How about here?
"""


cimport cython
from cython cimport numeric
cimport numpy as cnp
import numpy as np
cnp.import_array()

ctypedef void NpyIter
ctypedef struct NewNpyArrayIterObject:
    cnp.PyObject base
    NpyIter *iter

cdef void* nullptr

@cython.boundscheck(False)
cdef cnp.ndarray dimsimple(cnp.ndarray inarr,
                           double (*func)(double,void*),void* args=nullptr,
                           cnp.ndarray outarr=None):
    r"""dimsimple: Does Sphinx find this?

    :param inarr:
    :param func:
    :param args:
    :param outarr:
    :return:
    """
    cdef int j
    cdef cnp.ndarray[double] subinarr, suboutarr
    cdef void* cargs

    assert inarr.dtype==np.float
    if outarr is None:
        outarr=np.empty_like(inarr)
    cargs=<void*>args

    it = np.nditer([inarr,outarr], flags=['external_loop','buffered'],
                   op_flags=[['readonly'], ['writeonly']])
    for subinarr, suboutarr in it:
        for j in range(subinarr.shape[0]):
            suboutarr[j]=func(subinarr[j],cargs)
    return outarr

@cython.boundscheck(False)
cdef cnp.ndarray gridinput(cnp.ndarray inarr1, cnp.ndarray inarr2,
                           double (*func)(double,double,void*),void* args=nullptr,
                           cnp.ndarray outarr=None):

        cdef int j
        cdef cnp.ndarray[double] subinarr1,subinarr2,suboutarr
        cdef void* cargs

        assert inarr1.dtype==np.float
        assert inarr2.dtype==np.float
        inarr1,inarr2=np.meshgrid(inarr1,inarr2)
        if outarr is None:
            outarr = np.empty_like(inarr1)
        cargs=<void*>args

        it = np.nditer([inarr1,inarr2,outarr], flags=['external_loop','buffered'],
                       op_flags=[['readwrite'], ['readwrite'], ['readwrite']])
        for subinarr1,subinarr2,suboutarr in it:
            for j in range(subinarr1.shape[0]):
                suboutarr[j]=func(subinarr1[j],subinarr2[j],cargs)
        return outarr
