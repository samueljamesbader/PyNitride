# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:16:38 2017

@author: sam
"""
from poissolve.mesh import EpiStack, Mesh

# Put nothing before this
# because all other lines should be run *after* pytest.main
import pytest
if __name__=='__main__':
    pytest.main(args=[__file__,'--plots'])    
from poissolve.tests.runtests import plots

@plots
def test_mesh():
    #pass
    epistack=EpiStack(['GaN',2.5],['AlN',5],['GaN',17],['AlN',30])
    m=Mesh(epistack,max_dz=1,refinements=[[7.5,.1,1.1],[24,.25,1.2]])
    m.plot_mesh()
    assert 0

@plots
def test_mesh2():
    #pass
    epistack=EpiStack(['GaN',250],['AlN',5],['GaN',17],['AlN',30])
    m=Mesh(epistack,max_dz=1,refinements=[[7.5,.1,1.1],[24,.25,1.2]])
    m.plot_mesh()
    assert 0
