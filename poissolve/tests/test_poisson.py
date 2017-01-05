from poissolve.mesh_functions import MaterialFunction, PointFunction, MidFunction
from poissolve.solvers.poisson import PoissonSolver

# Put nothing before this
# because all other lines should be run *after* pytest.main
import pytest
if __name__=='__main__':
    #pytest.main(args=[__file__])
    pytest.main(args=[__file__,'--plots'])
from poissolve.tests.runtests import plots

@plots
def test_poisson(nonuniformmesh):
    #nonuniformmesh=uniformmesh
    nonuniformmesh.add_function('rho',PointFunction(nonuniformmesh))
    # Hackish addition of polarization
    P=MaterialFunction(nonuniformmesh,
            lambda mat: {
                "Gallium Nitride":5.6e-1,
                "Aluminum Nitride":0.0,
            }[mat['name']])
    #self._globalmesh.add_function('P',P)
    rho_p=P.differentiate(fill_value=0.0)
    nonuniformmesh.add_function('EF',PointFunction(nonuniformmesh,arr=0.0))
    nonuniformmesh.add_function('rho_p',rho_p)
    nonuniformmesh.add_function('rho',PointFunction(nonuniformmesh,rho_p.array.copy()))
    #nonuniformmesh.plot_function('rho_p')

    ps=PoissonSolver(nonuniformmesh)
    ps.solve()
    #nonuniformmesh.plot_function('mqV','.')

    mqV=nonuniformmesh['mqV']
    E=mqV.differentiate().array
    D=MidFunction(nonuniformmesh,MaterialFunction(nonuniformmesh,'eps').array*E)
    rho_calc=D.differentiate().array
    #print("rho_calc:")
    import numpy as np
    #print(nonuniformmesh['mqV'].array)
    assert np.allclose(rho_calc[1:-1],nonuniformmesh['rho'].array[1:-1])
    assert np.isclose(-D[-1],nonuniformmesh['rho'].array[-1])
