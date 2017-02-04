from pynitride.bandstuct.kp import kp_bandstructure
from pynitride import ParamDB, Material
pmdb=ParamDB(units='neu')

import pytest
if __name__=="__main__":
    #pytest.main([__file__])
    pass

def test_kp_bandstructure():
    E0=kp_bandstructure(Material("GaN"),[[0,0,0]],[0,0,0],spin_orbit=False)




if __name__=="__main__":
    test_kp_bandstructure()
