# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:16:38 2017

@author: sam
"""
import pytest
from poissolve.constants import eps_0

from pynitride.poissolve.mesh.structure import EpiStack, Mesh, Layer

if __name__=='__main__':
    #pytest.main(args=[__file__])
    pytest.main(args=[__file__,'--plots'])
from poissolve.tests.runtests import plots

def test_layer():
    l=Layer('layer1','GaN',10)
    assert l['eps']==10.6*eps_0

@plots
def test_mesh():
    #pass
    epistack=EpiStack(['GaN',2.5],['AlN',5],['GaN',17],['AlN',30])
    m=Mesh(epistack,max_dz=1,refinements=[[7.5,.1,1.1],[24,.25,1.2]])
    m.plot_mesh()

@plots
def test_mesh2():
    #pass
    epistack=EpiStack(['GaN',250],['AlN',5],['GaN',17],['AlN',30])
    m=Mesh(epistack,max_dz=1)
    print(m.z)
    m.plot_mesh()
    assert 0
