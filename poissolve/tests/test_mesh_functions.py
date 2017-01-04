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
    

def test_point_function():
    pass
def test_material_function(nonuniformmesh):
    F=MaterialFunction(nonuniformmesh,"eps")
    print(F)
    assert 0