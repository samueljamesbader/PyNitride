# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:05:16 2017

@author: sam
"""
import numpy as np
from pynitride.poissolve.maths.cfermidiracintegral import fd12,fd12p

# Put no pytest code before this: it should be run *after* pytest.main
import pytest
if __name__=="__main__":    
    if not pytest.main(args=[__file__]):
        numvalues=10000
        Nrepeat=100
        from timeit import timeit
        print("fd12 computed {:d} values in {:.3e} s (average of {:d} runs)."\
            .format(
                numvalues,
                timeit('fd12(x)','x=np.reshape(np.linspace(-10,10,{:d}),(2,{:d}))'.format(numvalues,int(numvalues/2)),
                    number=Nrepeat,globals=globals())/Nrepeat,
                Nrepeat))
        print("fd12p computed {:d} values in {:.3e} s (average of {:d} runs)."\
            .format(
                numvalues,
                timeit('fd12p(x)','x=np.reshape(np.linspace(-10,10,{:d}),(2,{:d}))'.format(numvalues,int(numvalues/2)),
                    number=Nrepeat,globals=globals())/Nrepeat,
                Nrepeat))
    

# How stringent a relative tolerance for comparison to reference values
rtols={'fd12':1e-5, 'fd12p':1e-3}

def test_fd12():
    
    # Give scalar, get float
    assert isinstance(fd12(0),float), "When supplied a scalar, fd12 doesn't return a float"

    # Give list, get ndarray
    #f=fd12([0])
    #assert isinstance(f,np.ndarray), "When supplied a list, fd12 should return an ndarray"
    #assert f.dtype=='float', "fd12's returned ndarray should be of dtype float"
    
    # Give ndarray, get ndarray
    f=fd12(fd12(np.array([0])))
    assert isinstance(f,np.ndarray), "When supplied an ndarray, fd12 should return an ndarray"
    assert f.dtype=='float', "fd12's returned ndarray should be of dtype float"

    # Check some values against the Fermi-Dirac integral calculator on Nanohub
    # Xingshu Sun; Mark Lundstrom; raseong kim (2014), "FD integral calculator,"
    # https://nanohub.org/resources/fdical. (DOI: 10.4231/D3SQ8QJ5T).
    assert np.allclose(
        fd12(np.array([-100,-10,0,10,100])),
        [3.720075976020836e-44,
         4.539920105264132e-05,
         7.651470246254078e-01,
         2.408465696463765e+01,
         7.523455915242853e+02],
        atol=0,rtol=rtols['fd12']),\
        "fd12 failed numerical comparison to Nanohub-computed values."

def test_fd12p():
    
    # Give scalar, get float
    assert isinstance(fd12p(0),float), "When supplied a scalar, fd12p doesn't return a float"

    # Give list, get ndarray
    #f=fd12p([0])
    #assert isinstance(f,np.ndarray), "When supplied a list, fd12p should return an ndarray"
    #assert f.dtype=='float', "fd12p's returned ndarray should be of dtype float"
    
    # Give ndarray, get ndarray
    f=fd12p(fd12(np.array([0])))
    assert isinstance(f,np.ndarray), "When supplied an ndarray, fd12p should return an ndarray"
    assert f.dtype=='float', "fd12p's returned ndarray should be of dtype float"

    # Check some values against the Fermi-Dirac integral calculator on Nanohub
    # Xingshu Sun; Mark Lundstrom; raseong kim (2014), "FD integral calculator,"
    # https://nanohub.org/resources/fdical. (DOI: 10.4231/D3SQ8QJ5T).
    assert np.allclose(
        fd12p(np.array([-100,-10,0,10,100])),
        [3.720075976020836e-44,
         4.539847236080550e-05,
         6.048986434216304e-01,
         3.552779239536616e+00,
         1.128332744278742e+01],
        atol=0,rtol=rtols['fd12p']),\
        "fd12p failed numerical comparison to Nanohub-computed values."