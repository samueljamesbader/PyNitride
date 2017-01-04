# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:47:33 2017

@author: sam
"""
import pytest
from poissolve.materials import Material, _materials

def test_material_get():
    m=Material("GaN")
    assert m['Eg']==_materials['GaN']['Eg'],\
        "Can't Material[][...] a single item from Material"
    assert m['dopants','Si','type']==_materials['GaN']['dopants']['Si']['type'],\
        "Can't Material[...] a multi-level item from Material"
    with pytest.raises(Exception): m['doesntexist'],\
        "Asking Material[...] for a nonexistant item doesn't raise an exception"
    assert m.get('doesntexist',7)==7,\
        "Material.get() with default fails"
    assert m.get(['dopants','Si','type'])==_materials['GaN']['dopants']['Si']['type'],\
        "Material.get() with multilevel request fails"
    

if __name__=='__main__':
    pytest.main(args=[__file__])