import numbers

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from pynitride.paramdb import kb,hbar,q,m_e, cm, pi
from pynitride.cython_maths import tdma, fd12, fd12p, idd,iddd
from pynitride.mesh import MaterialFunction, PointFunction, ConstantFunction, MidFunction, SubMesh
from pynitride.fem import assemble_stiffness_matrix, assemble_load_matrix, fem_solve, fem_get_error
from collections import OrderedDict
from operator import mul
from functools import reduce, lru_cache

from pynitride.visual import log, sublog




class PoissonSolver():
    r""" Solves the Poisson equation on a mesh.

    The boundary conditions assumed are that the potential is zero at the first mesh point, and the electric field
    goes to zero at the last mesh point (or, to be more precise, at the next midpoint after the last meshpoint).
    Two solve functions are available: :func:`~PoissonSolver.solve` and
    :func:`~pynitride.solvers.PoissonSolver.newton_step`.  The former is a direct solution, which can be
    obtained directly from charge integration.  The latter is a Newton-method solver appropriate for self-consistent
    iteration with a carrier solver.

    Args:
        mesh: the :class:`~pynitride.mesh.Mesh` on which to perform the solve
    """
    def __init__(self, mesh):
        m=self._mesh = mesh

        # Allocate output functions  on the mesh
        m.ensure_function_exists('rho',0)
        m.ensure_function_exists('rhoderiv',0)
        m.ensure_function_exists('fixedcharge',0)

        m.ensure_function_exists('Ndp',0)
        m.ensure_function_exists('Nam',0)
        m.ensure_function_exists('Ndpderiv',0)
        m.ensure_function_exists('Namderiv',0)

        m.ensure_function_exists('DP',0)
        m.ensure_function_exists('phi',0)


        # Collect lists of all defined dopants whether in mesh or not
        alldopants=sum((mb.matsys._dopants for mb in m._matblocks),[])
        self._donors    =[d for d in alldopants if d.endswith("Donor")]
        self._acceptors =[d for d in alldopants if d.endswith("Acceptor")]

        # Get surface barrier
        self._sbh=PoissonSolver.get_sbh(m)

        # Set the epsilon_factor to 1 (ie don't scale epsilon), and initialize bands
        self.update_epsfactor(epsfactor=1)
        self.update_bands_to_potential(m,0,sbh=self._sbh)

        # Assemble the load matrix
        self._load_matrix=assemble_load_matrix(m.ones_mid,m.dzp,n=1,
                               dirichelet1=True,dirichelet2=False)

    def update_epsfactor(self,epsfactor):
        """ Scales the epsilon used by this solver without actually changing epsilon on the mesh.

        The higher the `epsfactor`, the less coupling between `phi` and charge, iterative solutions are easier.
        The resulting solution is, of course, not correct for `epsfactor`!=1, but ramping `epsfactor` from a large
        value where the problem is easy down to 1 is a useful way to smoothly approach the solution.

        Args:
            epsfactor: the factor (generally >=1) by which epsilon should be scaled
        """
        self._eps=epsfactor*self._mesh.eps.copy()

        # Assemble stiffness from eps
        O=np.expand_dims(np.expand_dims(self._mesh.zeros_mid,0),0)
        eps=np.expand_dims(np.expand_dims(self._eps,0),0)
        self._stiffness_matrix= \
            assemble_stiffness_matrix(C0=O,Cl=None,Cr=None,
                                      C2=eps,dzp=self._mesh.dzp,
                                      dirichelet1=True,dirichelet2=False)

    @staticmethod
    def get_sbh(m):
        """ Get the surface barrier for a given mesh.

        If the mesh boundary is specified numerically, that's the surface barrier, otherwise asks the topmost
        material system for it's surface barrier given the mesh.

        Args:
            m: the global :class:`~pynitride.mesh.Mesh`

        Returns:
            the surface barrier (float)
        """
        surface=m._boundary[0]
        if isinstance(surface,numbers.Real):
            return surface
        else:
            return m._matblocks[0].matsys.surface_barrier(m._matblocks[0].mesh)

    def ionized_dopants(self,gotzloop=False):
        # Tiwari Compound Semiconductor Devices pg31-32
        m=self._mesh
        kT= kb * m.T

        m['Ndp']=0
        m['Ndpderiv']=0
        for d in self._donors:
            # TODO: Deal with the fact that idd only takes one g
            g=MaterialFunction(m,d+'g',default=0)[0]
            conc=MaterialFunction(m,d+'Conc',default=0)
            E=MaterialFunction(m,d+'E',default=0)
            eta=((m.EF.tmf()-m.Ec)+E)/kT
            m['Ndp']+=(conc*idd(eta,g)).tpf()
            m['Ndpderiv']+=(conc/kT*iddd(eta,g)).tpf()

        if gotzloop is False:
        #if 1:
            #m['Nam']=0
            Nam=PointFunction(m,value=0)
            m['Namderiv']=0
            for d in self._acceptors:
                # TODO: Deal with the fact that idd only takes one g
                g=MaterialFunction(m,d+'g',default=0)[0]
                conc=MaterialFunction(m,d+'Conc',default=0)
                E=MaterialFunction(m,d+'E',default=0)
                eta=((m.Ev-m.EF.tmf())+E)/kT
                #m['Nam']+=(conc*idd(eta,g)).tpf()
                Nam+=(conc*idd(eta,g)).tpf()
                m['Namderiv']-=(conc/kT*iddd(eta,g)).tpf()

            m['Nam']=Nam
        else:
            while True:
                Nam=PointFunction(m,value=0)
                m['Namderiv']=0
                for d in self._acceptors:
                    g=MaterialFunction(m,d+'g',default=0)[0]
                    conc=MaterialFunction(m,d+'Conc',default=0)
                    E=MaterialFunction(m,d+'E',default=0)
                    f=2.1828
                    gotz=m[d+'gotzshift']=-m.gotz*f*(q**2)/(4*pi*m.eps)*(m.Nam.tmf())**(1/3)
                    #gotz=0
                    eta=((m.Ev-m.EF.tmf())+E+gotz)/kT
                    Nam+=(conc*idd(eta,g)).tpf()
                    m['Namderiv']-=(conc/kT*iddd(eta,g)).tpf()
                    #log("Gotz max {:.6f}".format(float(np.max(np.abs(gotz)))),'debug')
                    #log("Namdiffmax {:.2e}".format(float(np.max(np.abs((Nam-m.Nam))))),'debug')
                if np.max(np.abs((Nam-m.Nam)))<1e16/cm**3:
                    m['Nam']=Nam
                    break
                m['Nam']=Nam



    def solve(self):
        r""" Solves the Poisson equation directly (not good for self-consistent looping)

        The equation is :math:`-\partial_z\epsilon\partial_z\phi=\rho`.  Do not use this function in
        a self-consistent poisson-carrier loop, because that's not super stable.  Instead
        use :func:`pynitride.solvers.PoissonSolver.newton_step`.

        """
        m=self._mesh

        # Dopants
        self.ionized_dopants()

        # Polarization
        P=MaterialFunction(m,'P',default=0)  # I don't like having to do this
        m.DP=-P.differentiate(fill_value=0)


        # Carriers
        p=m.p if ('p' in m) else 0
        n=m.n if ('n' in m) else 0

        # Total charge
        m.rho=p-n+m.Ndp-m.Nam+m.DP+m.fixedcharge

        # Solve and update
        fem_solve(self._stiffness_matrix,self._load_matrix,load_vec=m.rho,
                  val_out=m.phi,n=1,dirichelet1=True, dirichelet2=False)
        PoissonSolver.update_bands_to_potential(m,sbh=self._sbh)

    def newton_step(self, activation=1, doplot=False):
        r""" Solves the phi for one step of Newton iteration.

        The equation is
        :math:`-\left[\partial_z\epsilon\partial_z+\rho_0'\right]\delta\phi=\rho + \partial_z\epsilon\partial_z\phi_0`.

        """
        m=self._mesh

        # Dopants
        self.ionized_dopants()

        # Polarization
        P=MaterialFunction(m,'P',default=0)  # I don't like having to do this
        m.DP=-P.differentiate(fill_value=0)


        # Carriers
        p,pderiv=(m.p,m.pderiv) if ('p' in m) else (0,0)
        n,nderiv=(m.n,m.nderiv) if ('n' in m) else (0,0)

        # Total charge
        m.rho=p-n+m.Ndp-m.Nam+m.DP+m.fixedcharge
        m.rhoderiv=pderiv-nderiv+m.Ndpderiv-m.Namderiv

        # Note that the derivatives are with respect to the Fermi level at fixed band position, which
        # means it takes another negative sign to get the derivative with respect to potential
        drhodphi=-np.expand_dims(np.expand_dims(m.rhoderiv,0),0)

        # Assemble stiffness from eps
        eps=np.expand_dims(np.expand_dims(self._eps,0),0)
        stiffness_matrix= \
            assemble_stiffness_matrix(C0=-drhodphi,Cl=None,Cr=None,
                                      C2=eps,dzp=self._mesh.dzp,
                                      dirichelet1=True,dirichelet2=False)

        # The error is calculated using the direct solution stiffness_matrix (which depends only on epsilon,
        # not on the rho derivatives
        err0=fem_get_error(self._stiffness_matrix,self._load_matrix,load_vec=m.rho,test=m.phi,
                           err_out=None,n=1,dirichelet1=True,dirichelet2=False)

        # Solve and update
        dphi=self._recent_dphi=activation*fem_solve(stiffness_matrix,self._load_matrix,load_vec=err0,
                      val_out=None,n=1,dirichelet1=True, dirichelet2=False)
        PoissonSolver.update_bands_to_potential(m,phi=m.phi+dphi,sbh=self._sbh)

        return np.max(np.abs(dphi))#/activation

    @staticmethod
    def update_bands_to_potential(m,phi=None,sbh=None):
        """ Updates Ec, Ev, and phi to match the phi (potential) given.

        Args:
            m: the :class:`~pynitride.mesh.Mesh`
            phi: the new phi to use (MidFunction or scalar), None to just use current
            sbh: the surface potential, if not specified, will be calculated from the mesh
        """
        m.ensure_function_exists('phi',0)
        sbh=sbh if sbh is not None else PoissonSolver.get_sbh(m)
        if phi is not None: m['phi']=phi
        m['Ec']=-m.phi.tmf()+m.EF[0]+m.DE-m.DE[0]+m['Ec-E0']-m['Ec-E0'][0]+sbh
        m['Ev']=m.Ec-m.Eg
        m['E']=-m.phi.differentiate()

    def store_state(self):
        """ Stores the current phi in case we want to return to this current solution.

        Useful in iterative self-consistent solves for returning back to "the last point where things worked".
        See :func:`~pynitride.solvers.PoissonSolver.restore_state`.
        """
        self._storedphi=self._mesh.phi.copy()
    def restore_state(self):
        """ Restores the most recently saved phi.

        Useful in iterative self-consistent solves for returning back to "the last point where things worked".
        See :func:`~pynitride.solvers.PoissonSolver.store_state`.
        """
        PoissonSolver.update_bands_to_potential(self._mesh,self._storedphi,sbh=self._sbh)
    def shorten_last_step(self,factor):
        """ Shortens the `phi` step taken by the most recent solve by the given factor.

        Args:
            factor (float): the amount (0-1) by which the previous step should be rescaled
        """
        PoissonSolver.update_bands_to_potential(self._mesh,self._mesh.phi-(1-factor)*self._recent_dphi,sbh=self._sbh)

class Equilibrium():

    def __init__(self,mesh):
        self._mesh=mesh
        mesh.ensure_function_exists('EF',0)

    def solve(self):
        self._mesh['EF']=0

#class ChargeNeutral():
#
#    def __init__(self,mesh,carriersolvers=[],resolve_carriers=False):
#        self._mesh=mesh
#        self._mesh.ensure_function_exists('EF',value=0)
#        #self._mesh.ensure_function_exists('phi',0)
#        self._cs=carriersolvers
#        self._ps=PoissonSolver(mesh)
#        if resolve_carriers:
#            for cs in self._cs: cs.solve()
#
#    def solve(self, check='integrated', tol=None):
#        with sublog("Neutralizing charge","debug"):
#            m=self._mesh
#            if tol is None:
#                tol={'integrated':1e6/cm**2,'mean':1e9/cm**3}[check]
#            if check=='mean':
#                tol*=m.thickness
#            kT=k*np.max(m.T)
#            while True:
#                for cs in self._cs:
#                    cs.repopulate()
#                self._ps.ionized_dopants()
#                rho=(m.p-m.n+m.Ndp-m.Nam+m.DP).integrate(definite=True)
#                if abs(rho)<tol: break
#                rhoderiv=(m.pderiv-m.nderiv+m.Ndpderiv-m.Namderiv).integrate(definite=True)
#                log("Rho: {:.2e}        Rho' {:.2e}".format(float(rho),float(rhoderiv)),"debug")
#                dEF=np.sign(rho)*min(np.abs(rho/rhoderiv),kT)
#                m['EF']+=dEF


class Linear_Fermi():

    def __init__(self,mesh,contacts={'gate':0,'subs':-1}):
        self._mesh=mesh
        interfaces=[(0,None)]+mesh._interfacesp+[((len(mesh.zp)-1),None)]
        self._contacts=OrderedDict(sorted([(k,interfaces[v][0]) for k,v in contacts.items()],key=lambda x:x[1]))
        mesh['EF']=PointFunction(mesh)

    def solve(self,**voltages):
        lefts=list(self._contacts.items())[:-1]
        rights=list(self._contacts.items())[1:]
        for (clname,cl),(crname,cr) in zip(lefts,rights):
            l=-voltages.get(clname,0)
            r=-voltages.get(crname,0)
            self._mesh['EF'][cl:(cr+1)]=(self._mesh.zp[cl:(cr+1)]-self._mesh.zp[cl])/(self._mesh.zp[cr]-self._mesh.zp[cl])*(r-l)+l


class SelfConsistentLoop():
    def __init__(self,fieldsolvers=[],carriermodels=[]):
        self._fs=fieldsolvers
        self._cs=carriermodels

    def remove_carrier_model(self,cs):
        self._cs.remove(cs)
    def add_carrier_model(self,cs):
        self._cs.append(cs)
    def swap_carrier_model(self,remove,add):
        self.remove_carrier_model(remove)
        self.add_carrier_model(add)
    def isolve_fields(self, activation=1):
        return sum(fs.newton_step(activation=activation) for fs in self._fs)
    def solve_fields(self):
        return [fs.solve() for fs in self._fs]
    def solve_carriers(self):
        [cs.solve_and_repopulate() for cs in self._cs]

    def loop(self, tol=1e-5, max_iter=100, min_activation=.05,init_activation=1):
        adec=2
        with sublog("Starting SC loop"):
            err=np.inf
            i=0
            a=init_activation
            while err/a>tol:
                if i>=max_iter:
                    raise Exception("Maximum iteration reached in SC loop")
                self.solve_carriers()
                errprev=err
                err=self.isolve_fields(activation=a)
                log("iter: {:3d}  err: {:.2e}  activ: {:g}".format(i,err,a))
                while err>errprev:
                    a/=adec
                    log("Retrying with Poisson activation={:g}".format(a))
                    if a<min_activation:
                        raise Exception("Couldn't reduce error in SC loop")
                    for fs in self._fs:fs.shorten_last_step(1/adec)
                    self.solve_carriers()
                    err=self.isolve_fields(activation=a)
                    log("       iter: {:3d}  err: {:.2e}".format(i,err))
                a=min(1.2*a,1)
                i+=1
            #log("Post-solve")
            #self.solve_fields()
            log("Loop finished in {:2d} iterations with err={:g}".format(i,err))

    def ramp_epsfactor(self, start=1e4, stop=1, dlefstart=.1, dlefmax=.5, dlefmin=.005, max_iter=20, tol=1e-5, min_activation=.1):
        with sublog("Starting eps factor ramp from {:g} to {:g}".format(start,stop)):
            lefstart=np.log10(start)
            lefstop=np.log10(stop)
            lef=lefstart

            dlefstart*=np.sign(lefstop-lefstart)
            dlef=dlefstart
            prevlef=None
            while True:
                ef=10**lef
                log("Eps factor: {:.2e}".format(ef))

                for fs in self._fs:
                    fs.store_state()
                    fs.update_epsfactor(ef)
                    self.isolve_fields()
                try:
                    self.loop(max_iter=max_iter, tol=tol, min_activation=min_activation)
                    if (lef-lefstop)<1e-9:
                        break

                    # Next lef
                    prevlef=lef
                    dlef=np.sign(dlef)*min(np.abs(2*dlef),np.abs(dlefmax))
                    lef=lef+dlef
                    # if we passed the end, go back
                    if np.sign(lef-lefstop)!=np.sign(lefstart-lefstop): lef=lefstop
                except Exception as e:
                    log("Failure: {}".format(str(e)))
                    if prevlef is None:
                        raise Exception("Failed at initial epsfactor")
                    ef=10**prevlef
                    log("Restoring at {:.2e}".format(ef))
                    for fs in self._fs:
                        fs.restore_state()
                        fs.update_epsfactor(ef)
                    self.solve_carriers()

                    dlef=dlef/2
                    if np.abs(dlef)<np.abs(dlefmin):
                        raise Exception("Eps factor step size too small")
                    lef=prevlef+dlef
                    # if we passed the end, go back
                    if np.sign(lef-lefstop)!=np.sign(lefstart-lefstop): lef=lefstop
            log("Done eps factor ramp")

    def ramp_param(self, name, xstart, xstop, dxstart=.1, dxmax=.5, dmin=.005,
            max_iter=20, tol=1e-5, min_activation=.1):
        with sublog("Starting {} ramp from {:g} to {:g}".format(name,start,stop)):
            x=xstart

            dxstart*=np.sign(xstop-xstart)
            dx=dxstart
            prevx=None
            while True:
                log("{}: {:.2g}".format(name,x))

                for fs in self._fs:
                    fs.store_state()
                    fs.update_epsfactor(ef)
                    self.isolve_fields()
                try:
                    self.loop(max_iter=max_iter, tol=tol, min_activation=min_activation)
                    if (x-xstop)<1e-9:
                        break

                    # Next x
                    prevx=x
                    dx=np.sign(dx)*min(np.abs(2*dx),np.abs(dxmax))
                    x=x+dx
                    # if we passed the end, go back
                    if np.sign(x-xstop)!=np.sign(xstart-xstop): x=xstop
                except Exception as e:
                    log("Failure: {}".format(str(e)))
                    if prevx is None:
                        raise Exception("Failed at initial epsfactor")
                    x=prevx
                    log("Restoring at {:.2e}".format(x))
                    for fs in self._fs:
                        fs.restore_state()
                        fs.update_epsfactor(x)
                    self.solve_carriers()

                    dx=dx/2
                    if np.abs(dx)<np.abs(dx):
                        raise Exception("Eps factor step size too small")
                    x=prevx+dx
                    # if we passed the end, go back
                    if np.sign(x-xstop)!=np.sign(xstart-xstop): x=xstop
            log("Done {} ramp".format(name))
