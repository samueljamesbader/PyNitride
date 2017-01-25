import numpy as np
from pynitride.poissolve.mesh.structure import Mesh, SubMesh
from pynitride.poissolve.mesh.functions import PointFunction
from pynitride.poissolve.solvers.poisson import PoissonSolver
from pynitride.poissolve.solvers.schrodinger import SchrodingerSolver
from pynitride.poissolve.solvers.fermidirac import FermiDirac3D

class Coupled_FD_Poisson():

    def __init__(self,mesh):
        m=mesh
        # Set some stuff
        m['EF']=PointFunction(m,0.0)
        if 'rho_pol' not in m:
            m['rho_pol']=PointFunction(m,0.0)
        if 'rho' not in m:
            m['rho']=PointFunction(m,0.0)
        if 'arho2' not in m:
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



class Coupled_Schrodinger_Poisson():

    def __init__(self,mesh, carriers=['electron','hole'],schrodinger=None):
        m=mesh
        # Set some stuff
        m['EF']=PointFunction(m,0.0)
        if 'rho_pol' not in m:
            m['rho_pol']=PointFunction(m,0.0)
        if 'rho' not in m:
            m['rho']=PointFunction(m,0.0)
        if 'arho2' not in m:
            m['arho2']=PointFunction(m,0.0) # is this necessary?



        self._classical_charge_solvers=[FermiDirac3D(m)]

        schrofull=(schrodinger is None)
        if schrofull: schrodinger=m
        self._quantum_charge_solvers=[SchrodingerSolver(schrodinger,carriers=carriers)]
        if not schrofull:
            if schrodinger._slice.start is not None and schrodinger._slice.start>0:
                fd_sm=SubMesh(m,None,schrodinger._slice.start)
                fd=FermiDirac3D(fd_sm)
                self._quantum_charge_solvers+=[fd]
            if schrodinger._slice.stop is not None and schrodinger._slice.stop<len(m.z):
                fd_sm=SubMesh(m,schrodinger._slice.stop,None)
                fd=FermiDirac3D(fd_sm)
                self._quantum_charge_solvers+=[fd]

        # Prep solvers
        self._ps=PoissonSolver(m)


    def solve(self, low_act=4, rise=500, tol=1e-8, max_iter=100, callback=lambda *args: None):
        self._ps.solve()
        if callback(): return
        for activation in np.logspace(-low_act,-0.,rise):
            #self._fd.solve(activation=activation)
            for s in self._classical_charge_solvers: s.solve(activation)
            err=self._ps.isolve(visual=False)
            if callback(): return
        for i in range(max_iter):
            for s in self._classical_charge_solvers: s.solve(activation=1)
            err=self._ps.isolve(visual=False)
            if callback(): return
            if err<tol:
                print("Semi-classical success (max err={:.2g})after {:d} refinement iterations".format(err,i-1))
                break
        assert err<tol, "Stopped because reached max_iter with err ({:.2g}) > tol ({:.2g}).".format(err,tol)
        for i in range(max_iter):
            for s in self._quantum_charge_solvers: s.solve(activation=1)
            err=self._ps.isolve(visual=False)
            if callback(): return
            if err<tol:
                print("Full success (max err={:.2g})after {:d} refinement iterations".format(err,i-1))
                break
        assert err<tol, "Stopped because reached max_iter with err ({:.2g}) > tol ({:.2g}).".format(err,tol)
