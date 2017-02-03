# -*- coding: utf-8 -*-
r""" Tests the high-performance mathematical methods in :py:mod:`pynitride.poissolve.maths`."""
import numpy as np
from pynitride.poissolve.maths import fd12,fd12p, tdma
from scipy.sparse import diags
from timeit import timeit
import pytest
if __name__=="__main__": pytest.main(args=[__file__,'-s'])

# How stringent a relative tolerance for comparison to reference values
rtols={'fd12':1e-4, 'fd12p':1e-3}

def test_fd12():
    r"""Spot-checks :py:func:`~pynitride.poissolve.maths.fd12` against a couple Nanohub-computed values."""
    
    # Give list, get ndarray
    f=fd12([0])
    assert isinstance(f,np.ndarray), "When supplied a list, fd12 should return an ndarray"
    assert f.dtype=='float', "fd12's returned ndarray should be of dtype float"
    
    # Give ndarray, get ndarray
    f=fd12(fd12(np.array([0])))
    assert isinstance(f,np.ndarray), "When supplied an ndarray, fd12 should return an ndarray"
    assert f.dtype=='float', "fd12's returned ndarray should be of dtype float"

    # Check some values against the Fermi-Dirac integral calculator on Nanohub
    # Xingshu Sun; Mark Lundstrom; raseong kim (2014), "FD integral calculator,"
    # https://nanohub.org/resources/fdical. (DOI: 10.4231/D3SQ8QJ5T).
    assert np.allclose(
        fd12(np.array([-100,-10,0,5,10,100])),
        [3.720075976020836e-44,
         4.539920105264132e-05,
         7.651470246254078e-01,
         8.844208895242954e+00,
         2.408465696463765e+01,
         7.523455915242853e+02],
        atol=0,rtol=rtols['fd12']),\
        "fd12 failed numerical comparison to Nanohub-computed values."

def test_fd12p():
    r"""Spot-checks :py:func:`~pynitride.poissolve.maths.fd12p` against a couple Nanohub-computed values."""

    # Give list, get ndarray
    f=fd12p([0])
    assert isinstance(f,np.ndarray), "When supplied a list, fd12p should return an ndarray"
    assert f.dtype=='float', "fd12p's returned ndarray should be of dtype float"
    
    # Give ndarray, get ndarray
    f=fd12p(fd12(np.array([0])))
    assert isinstance(f,np.ndarray), "When supplied an ndarray, fd12p should return an ndarray"
    assert f.dtype=='float', "fd12p's returned ndarray should be of dtype float"

    # Check some values against the Fermi-Dirac integral calculator on Nanohub
    # Xingshu Sun; Mark Lundstrom; raseong kim (2014), "FD integral calculator,"
    # https://nanohub.org/resources/fdical. (DOI: 10.4231/D3SQ8QJ5T).
    assert np.allclose(
        fd12p(np.array([-100,-10,0,5,10,100])),
        [3.720075976020836e-44,
         4.539847236080550e-05,
         6.048986434216304e-01,
         2.472987622482944e+00,
         3.552779239536616e+00,
         1.128332744278742e+01],
        atol=0,rtol=rtols['fd12p']),\
        "fd12p failed numerical comparison to Nanohub-computed values."

def test_fd_timing(capfd):
    r"""Tests how fast the :py:func:`~pynitride.poissolve.maths.fd12` and :py:func:`~pynitride.poissolve.maths.fd12`
    compute large vector inputs."""
    numvalues=10000
    Nrepeat=100
    fd12_timing=timeit('fd12(x)','x=np.reshape(np.linspace(-10,10,{:d}),(2,{:d}))' \
       .format(numvalues,int(numvalues/2)),number=Nrepeat,globals=globals())/Nrepeat
    fd12p_timing=timeit('fd12p(x)','x=np.reshape(np.linspace(-10,10,{:d}),(2,{:d}))' \
        .format(numvalues,int(numvalues/2)),number=Nrepeat,globals=globals())/Nrepeat
    assert fd12_timing  <6e-3, "Fermi-Dirac 1/2 integral is running slow..."
    assert fd12p_timing <6e-3, "Fermi-Dirac -1/2 integral is running slow..."
    if 1:#with capfd.disabled():
        print("\n")
        print("fd12 computed {:d} values in {:.3e} s (average of {:d} runs)." \
            .format(numvalues,fd12_timing,Nrepeat))
        print("fd12p computed {:d} values in {:.3e} s (average of {:d} runs)." \
            .format(numvalues,fd12p_timing,Nrepeat))



def generate_tridiagonal_problem(N):
    r"""Generates a random tridiagonal matrix problem (``a,b,c,d``) for :py:func:`~pynitride.poissolve.maths.tdma`.

    The off-diagonals are limited to half of the diagonal in the same row, which guarantees diagonal dominance as
    required by `CFD-Online <https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)>`_.

    :return: a tuple of ``a,b,c,d``.
    """

    b=-np.random.rand(N)
    a=-np.random.rand(N)*.5*b
    c=-np.random.rand(N)*.5*b
    d=np.random.rand(N)
    a[0]=0
    c[N-1]=0

    return a,b,c,d

def test_tdma():
    r""" Tests :py:func:`~pynitride.poissolve.maths.tdma` numerically for a random tridiagonal problem."""
    a,b,c,d=generate_tridiagonal_problem(100)
    x=tdma(a,b,c,d)
    A=diags([a[1:],b,c[:-1]],offsets=[-1,0,1])
    b=A@x

    assert np.allclose(b,d)

def test_tdma_timing():
    r""" Tests how fast :py:func:`~pynitride.poissolve.maths.tdma` computes a large problem."""
    numvalues=10000
    Nrepeat=100
    tdma_timing=timeit('tdma(a,b,c,d)',"a,b,c,d=generate_tridiagonal_problem(10000)",
                       number=Nrepeat,globals=globals())/Nrepeat
    print("\ntdma computed {:d} values in {:.3e} s (average of {:d} runs)." \
          .format(numvalues,tdma_timing,Nrepeat))

    # TODO: add assertion about tdma speed
