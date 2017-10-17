#from pynitride.bandstuct.kp import kp_6x6
from pynitride.bandstruct.reciprocal import get_symmetry_point
from pynitride import ParamDB, Material
import numpy as np
pmdb=ParamDB(units='neu')

import pytest
if __name__=="__main__":
    pytest.main([__file__,'-s'])

def test_symmetrypoints():
    r""" Tests that the symmetry points Gamma and A are correct for GaN."""
    GaN=Material("GaN",pmdb=pmdb)
    assert np.allclose(get_symmetry_point("Gamma",GaN),[0,0,0])
    assert np.allclose(get_symmetry_point("A",GaN),[0,0,np.pi/GaN['lattice.c']])
