import numpy as np
from poissolve.mesh.functions import PointFunction
from poissolve.solvers.poisson import PoissonSolver
from poissolve.solvers.fermidirac import FermiDirac3D

class Coupled_FD_Poisson():

    def __init__(self,mesh):
        m=mesh
        # Set some stuff
        m['EF']=PointFunction(m,0.0)
        if 'rho_pol' not in m:
            m['rho_pol']=PointFunction(m,0.0)
        m['rho']=PointFunction(m,0.0)
        m['arho2']=PointFunction(m,0.0) # is this necessary?

        # Prep solvers
        self._ps=PoissonSolver(m)
        self._fd=FermiDirac3D(m)


    def solve(self, low_act=4, rise=500, tol=1e-8, max_iter=100, callback=lambda *args: None):
        self._ps.solve()
        if callback(): return
        for activation in np.logspace(-low_act,-0.,rise):
            self._fd.solve(activation=activation)
            err=self._ps.isolve(visual=False)
            if callback(): return
        for i in range(max_iter):
            self._fd.solve(activation=1)
            err=self._ps.isolve(visual=False)
            if callback(): return
            if err<tol:
                print("Success (max err={:.2g})after {:d} refinement iterations".format(err,i-1))
                break
        assert err<tol, "Stopped because reached max_iter with err ({:.2g}) > tol ({:.2g}).".format(err,tol)
