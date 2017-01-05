import numpy as np
import matplotlib.pyplot as mpl
from poissolve.constants import cm,nm
from poissolve.mesh_functions import MaterialFunction, PointFunction, MidFunction, RegionFunction
from poissolve.solvers.poisson import PoissonSolver
from poissolve.solvers.fermi_dirac import FermiDirac3D
from poissolve.mesh import EpiStack, Mesh

# Put nothing before this
# because all other lines should be run *after* pytest.main
import pytest
if __name__=='__main__':
    #pytest.main(args=[__file__])
    pytest.main(args=[__file__,'--plots'])
from poissolve.tests.runtests import plots

@plots
def test_fermi_dirac(nonuniformmesh):
    xp=100*nm
    xn=100*nm
    Nd=1e18*cm**-3
    Na=1e18*cm**-3
    class X: pass
    self=X()
    epistack = EpiStack(['pGaN', 'GaN', xp], ['nGaN', 'GaN', xn])
    self._mesh = Mesh(epistack, max_dz=1, refinements=[[xp, .05, 1.3]])

    self._mesh.add_function('rho_p', PointFunction(self._mesh, arr=0.0))

    self._mesh.add_function('SiActiveConc', RegionFunction(self._mesh,
                                                           lambda name: (name == "nGaN") * Nd, pos='point'))
    self._mesh.add_function('SiIonizedConc', PointFunction(self._mesh))
    self._mesh.add_function('MgActiveConc', RegionFunction(self._mesh,
                                                           lambda name: (name == "pGaN") * Na, pos='point'))
    self._mesh.add_function('MgIonizedConc', PointFunction(self._mesh))

    self._mesh.add_function('EF', PointFunction(self._mesh, arr=0.0))
    self._mesh.add_function('rho', PointFunction(self._mesh, arr=0.0))
    self._arho2 = self._mesh.add_function('arho2', PointFunction(self._mesh, arr=0.0))
    self._ps = PoissonSolver(self._mesh)
    self._fd = FermiDirac3D(self._mesh)
    self._fd.solve()

@plots
def test_fermi_dirac():
    xp=100*nm
    xn=100*nm
    Nd=1e18*cm**-3
    Na=1e18*cm**-3
    epistack = EpiStack(['nGaN', 'GaN', xp], ['nGaN', 'GaN', xn])
    mesh = Mesh(epistack, max_dz=.01*nm)

    mesh.add_function('rho_p', PointFunction(mesh, arr=0.0))

    mesh.add_function('SiActiveConc', RegionFunction(mesh,lambda name: (name == "nGaN") * Nd, pos='point'))
    mesh.add_function('SiIonizedConc', PointFunction(mesh))
    mesh.add_function('MgActiveConc', RegionFunction(mesh,lambda name: (name == "pGaN") * Na, pos='point'))
    mesh.add_function('MgIonizedConc', PointFunction(mesh))

    mesh.add_function('EF', PointFunction(mesh, arr=0.0))
    mesh.add_function('rho', PointFunction(mesh, arr=0.0))
    mesh.add_function('Ec', PointFunction(mesh, arr=np.linspace(-1,4,len(mesh.z))))
    mesh.add_function('Ev', PointFunction(mesh,arr=(mesh['Ec'].array-MaterialFunction(mesh,"Eg").to_point_function().array)))

    fd = FermiDirac3D(mesh)
    fd.solve()

    rho=mesh['rho'].array
    Ec=mesh['Ec'].array
    mpl.figure()
    mpl.plot(Ec,np.abs(rho)*(rho>0))
    mpl.plot(Ec,np.abs(rho)*(rho<0))
    mpl.yscale('log')

    mpl.figure()
    from scipy import gradient
    drhodEc=gradient(rho,Ec[1]-Ec[0])
    rhoderiv=mesh['rhoderiv']
    mpl.plot(Ec,drhodEc,'.')
    mpl.plot(Ec,rhoderiv)
    assert np.allclose(drhodEc[1:-1],rhoderiv[1:-1],atol=1e-5,rtol=1e-3)

