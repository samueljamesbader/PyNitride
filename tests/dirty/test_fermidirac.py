import matplotlib.pyplot as mpl
import numpy as np
import pytest
from poissolve.constants import cm,nm
from poissolve.mesh.functions import MaterialFunction, PointFunction, RegionFunction
from poissolve.solvers.fermidirac import FermiDirac3D
from poissolve.solvers.poisson import PoissonSolver

from pynitride.poissolve.mesh import EpiStack, Mesh

if __name__=='__main__':
    #pytest.main(args=[__file__])
    pytest.main(args=[__file__,'--plots'])
from poissolve.tests.runtests import plots

@plots
def test_fermi_dirac(nonuniformmesh):
    return
    xp=100*nm
    xn=100*nm
    Nd=1e18*cm**-3
    Na=1e18*cm**-3
    class X: pass
    self=X()
    epistack = EpiStack(['pGaN', 'GaN', xp], ['nGaN', 'GaN', xn])
    self._mesh = Mesh(epistack, max_dz=1, refinements=[[xp, .05, 1.3]])

    self._mesh['rho_pol']=PointFunction(self._mesh, arr=0.0)

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

    mesh['rho_pol']=PointFunction(mesh, 0.0)

    mesh['SiActiveConc']= RegionFunction(mesh,lambda name: (name == "nGaN") * Nd, pos='point')
    mesh['MgActiveConc']= RegionFunction(mesh,lambda name: (name == "pGaN") * Na, pos='point')

    mesh['EF']=PointFunction(mesh, 0.0)
    mesh['rho']=PointFunction(mesh, 0.0)
    mesh['Ec']=PointFunction(mesh, np.linspace(-1, 4, len(mesh.zp)))
    mesh['Ev']=PointFunction(mesh,(mesh['Ec']-MaterialFunction(mesh,"Eg").to_point_function()))

    fd = FermiDirac3D(mesh)
    fd.solve()

    rho=mesh['rho']
    Ec=mesh['Ec']
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

