# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 23:25:46 2017

@author: sam
"""
from poissolve.mesh_functions import MaterialFunction

# Put no pytest code before this: it should be run *after* pytest.main
import pytest
if __name__=='__main__':
    pytest.main(args=[__file__])

def test_function_indexing(nonuniformmesh):
    Fmid=MaterialFunction(nonuniformmesh,"eps")
    print(Fmid[1])
    print(Fmid[1:10]._z)
    assert 0

def test_posconversion(nonuniformmesh):
    return
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
