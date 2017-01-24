import numpy as np
from poissolve.maths.tdma import tdma
from scipy.sparse import diags
import pytest
from timeit import timeit

def generate_tridiagonal_problem(N):

    b=-np.random.rand(N)
    a=-np.random.rand(N)*.5*b
    c=-np.random.rand(N)*.5*b
    d=np.random.rand(N)
    a[0]=0
    c[N-1]=0

    return a,b,c,d

if __name__=='__main__':
    if not pytest.main(args=[__file__]):
        numvalues=10000
        Nrepeat=100
        from timeit import timeit
        print("tdma computed {:d} values in {:.3e} s (average of {:d} runs)." \
            .format(
            numvalues,
            timeit('tdma(a,b,c,d)',"a,b,c,d=generate_tridiagonal_problem(10000)",
                   number=Nrepeat,globals=globals())/Nrepeat,
            Nrepeat))

def test_tdma():
    a,b,c,d=generate_tridiagonal_problem(100)
    x=tdma(a,b,c,d)
    A=diags([a[1:],b,c[:-1]],offsets=[-1,0,1])
    b=A@x

    assert np.allclose(b,d)
