# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 00:43:13 2017

@author: sam
"""
import pytest

def pytest_addoption(parser):
    parser.addoption("--plots", action="store_true",
        help="run tests including plot generation")
        

@pytest.fixture(scope='session',autouse=True)
def clear_plots():
    if pytest.config.getoption("--plots"):
        print('getting executed')
        import matplotlib.pyplot as mpl
        mpl.close('all')
        yield
        mpl.show()
    else: yield
        
@pytest.fixture(scope='module')  
def nonuniformmesh():
    from poissolve.mesh import EpiStack, Mesh
    epistack=EpiStack(['GaN',2.5],['AlN',5],['GaN',17],['AlN',30])
    m=Mesh(epistack,max_dz=1,refinements=[[7.5,.1,1.1],[24,.25,1.2]])
    return m