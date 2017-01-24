import pytest

from poissolve.mesh.functions import MaterialFunction, PointFunction, MidFunction
from poissolve.solvers.poisson import PoissonSolver

if __name__=='__main__':
    #pytest.main(args=[__file__])
    pytest.main(args=[__file__,'--plots'])
from poissolve.tests.runtests import plots

@plots
def test_poisson(nonuniformmesh):
    #nonuniformmesh=uniformmesh
    nonuniformmesh['rho']=PointFunction(nonuniformmesh)
    # Hackish addition of polarization
    P=MaterialFunction(nonuniformmesh,
            lambda mat: {
                "Gallium Nitride":5.6e-1,
                "Aluminum Nitride":0.0,
            }[mat['name']])
    #self._globalmesh['P',P)
    rho_p=P.differentiate(fill_value=0.0)
    nonuniformmesh['EF']=PointFunction(nonuniformmesh,0.0)
    nonuniformmesh['rho_p']=rho_p
    nonuniformmesh['rho']=PointFunction(nonuniformmesh,rho_p.copy())
    nonuniformmesh.plot_function('rho_p')

    ps=PoissonSolver(nonuniformmesh)
    ps.solve()
    nonuniformmesh.plot_function('mqV','.')

    mqV=nonuniformmesh['mqV']
    E=mqV.differentiate()
    D=MidFunction(nonuniformmesh,MaterialFunction(nonuniformmesh,'eps')*E)
    rho_calc=D.differentiate()
    print("rho_calc:")
    import numpy as np
    print(nonuniformmesh['mqV'])
    assert np.allclose(rho_calc[1:-1],nonuniformmesh['rho'][1:-1])
    assert np.isclose(-D[-1],nonuniformmesh['rho'][-1])
    import matplotlib.pyplot as mpl
    mpl.show()
