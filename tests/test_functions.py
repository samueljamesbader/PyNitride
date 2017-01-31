# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 23:25:46 2017

@author: sam
"""
import numpy as np
import pytest
from poissolve.mesh.functions import MaterialFunction, MidFunction, PointFunction

from pynitride.poissolve.mesh import EpiStack, Mesh

if __name__=='__main__':
    pytest.main(args=[__file__])

def test_function_indexing():
    xp = 5
    xn = 5
    m = Mesh(EpiStack(['pGaN', 'GaN', 5], ['nGaN', 'GaN', 5]), max_dz=1)  # ,refinements=[[xp,.1,1.3]])
    sm = m.submesh([3, 8])


    pf = PointFunction(m, value=np.arange(11))
    assert np.allclose(pf,np.arange(11))
    spf=pf.restrict(sm)
    assert np.allclose(spf,np.arange(3,8))
    spf[:]=0
    assert np.allclose(pf,[0,1,2,0,0,0,0,0,8,9,10])
    assert np.allclose(pf.integrate().differentiate(),[ np.NaN,   1.,   2.,   0.,   0.,   0.,   0.,   0.,   8.,   9.,  np.NaN], equal_nan=True)
    assert np.allclose(pf.differentiate().integrate(flipped=True),[-10.,  -9.,  -8., -10., -10., -10., -10., -10.,  -2.,  -1.,   0.])
    MidFunction(m,m._dz).to_point_function()


def test_posconversion(nonuniformmesh):
    Fmid=MaterialFunction(nonuniformmesh,"eps")
    Fpt=Fmid.to_point_function()
    for (i,ll,lr),(il,ir,ll,lr) in zip(nonuniformmesh.interfaces_point,nonuniformmesh.interfaces_mid):
        print("Z")
        print(nonuniformmesh.z[i])
        print(nonuniformmesh.zp[il],nonuniformmesh.zp[ir])
        print("F")
        print(Fpt[i])
        print(Fmid[il],Fmid[ir])
        print(Fmid[il-1],Fmid[ir+1])
        assert Fpt[i]==(Fmid[il]+Fmid[ir])/2, "Default (unweighted) mid -> point function conversion failed"
