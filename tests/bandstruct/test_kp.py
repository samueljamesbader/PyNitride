from pynitride.bandstruct.kp import kp_6x6
from pynitride import ParamDB, Material
import numpy as np
pmdb=ParamDB(units='neu')

import pytest
if __name__=="__main__":
    pytest.main([__file__,'-s'])

def test_kp_bandstructure():
    GaN=Material("GaN")
    ez=.01394
    et=-.02415
    E0=kp_6x6(GaN, [[0, 0, 0]], [0, 0, 0], spin_orbit=False)[0]
    Es=kp_6x6(GaN, [[0, 0, 0]], [et, et, ez], spin_orbit=False)[0]

    # Without SO splitting, conduction band moves as ac1*ez+2*ac2*et, where ac1=a1+D1, ac2=a2+D2b
    CB_move=Es[6]-E0[6]
    assert np.isclose(CB_move,(GaN['kp.a1']+GaN['kp.D1'])*ez+2*(GaN['kp.a2']+GaN['kp.D2'])*et)

    # Without SO splitting, HH moves as (D1+D3)*ez+(D2+D4)*et
    HH_move=Es[5]-E0[5]
    assert np.isclose(HH_move,(GaN['kp.D1']+GaN['kp.D3'])*ez+2*(GaN['kp.D2']+GaN['kp.D4'])*et)

    # Without SO splitting, LH moves as (D1+D3)*ez+(D2+D4)*et
    LH_move=Es[3]-E0[3]
    assert np.isclose(LH_move,(GaN['kp.D1']+GaN['kp.D3'])*ez+2*(GaN['kp.D2']+GaN['kp.D4'])*et)

    # Without SO splitting, CH moves as (D1)*ez+(D2)*et
    CH_move=Es[1]-E0[1]
    assert np.isclose(CH_move,(GaN['kp.D1'])*ez+2*(GaN['kp.D2'])*et)




if __name__=="__main__":
    test_kp_bandstructure()
