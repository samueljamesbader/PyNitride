import numbers

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from pynitride import kb,hbar,q,m_e, cm, pi
from pynitride.core.cython_maths import tdma, fd12, fd12p, idd,iddd
from pynitride import MaterialFunction, NodFunction,  MidFunction, SubMesh
from pynitride.core.fem import assemble_stiffness_matrix, assemble_load_matrix, fem_solve, fem_get_error
from collections import OrderedDict
from operator import mul
from functools import reduce, lru_cache

from pynitride import log, sublog




class PoissonSolver():
    r""" Solves the Poisson equation on a mesh.

    The boundary conditions assumed are Dirichelet at the surface (ie fixed surface barrier) and Neumann at the bottom
    (ie thick substrate).  Accounts for charge from (1) carriers which are filled into `mesh['n']` and `mesh['p']` by
    some Carrier Model (2) polarization which is filled into `mesh['P']` (generally by the `MaterialSystem`),
    differentiating that to fill `mesh['DP']`, and (3) dopants which for which the values are drawn from the mesh (see
    :func:`~pynitride.solvers.PoissonSolver.ionized_dopants`).

    Two solve functions are available: :func:`~PoissonSolver.solve` and
    :func:`~pynitride.solvers.PoissonSolver.newton_step`.  The former is a direct solution, which can be
    obtained directly from charge integration.  The latter is a Newton-method solver appropriate for self-consistent
    iteration with a carrier solver. For use with this method, the carrier models must implement not just `n` and `p`
    but also `nderiv` and `pderiv` to supply the relevant derivative information.

    A static convenience function :func:`~pynitride.solvers.PoissonSolver.update_bands_to_potential`
    is also available to set bands for simulations where the Poisson equation is not needed.

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

    def ionized_dopants(self):
        """ Computes the ionized dopant densities and their derivatives

        Fills the values `Ndp`, `Ndpderiv`, `Nam`, `Namderiv` onto the mesh
        for the total ionized donor and acceptor densities and derivatives.

        """
        # Tiwari Compound Semiconductor Devices pg 31-32
        m=self._mesh
        kT= kb * m.T

        m['Ndp']=0
        m['Ndpderiv']=0
        for d in self._donors:
            g=MaterialFunction(m,d+'g',default=0)
            conc=MaterialFunction(m,d+'Conc',default=0)
            E=MaterialFunction(m,d+'E',default=0)

            eta=((m.EF.tmf()-m.Ec)+E)/kT
            m['Ndp']+=(conc*idd(eta,g)).tpf()
            m['Ndpderiv']+=(conc/kT*iddd(eta,g)).tpf()

        m['Nam']=0
        m['Namderiv']=0
        for d in self._acceptors:
            g=MaterialFunction(m,d+'g',default=0)
            conc=MaterialFunction(m,d+'Conc',default=0)
            E=MaterialFunction(m,d+'E',default=0)

            eta=((m.Ev-m.EF.tmf())+E)/kT
            m['Nam']+=(conc*idd(eta,g)).tpf()
            m['Namderiv']-=(conc/kT*iddd(eta,g)).tpf()


    def solve(self):
        r""" Solves the Poisson equation directly (not good for self-consistent looping)

        The equation is :math:`-\partial_z\epsilon\partial_z\phi=\rho`.  Do not use this function in
        a self-consistent poisson-carrier loop, because that's not super stable.  Instead
        use :func:`~pynitride.solvers.PoissonSolver.newton_step`.

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

    def newton_step(self, activation=1):
        r""" Solves the phi for one step of Newton iteration.

        The equation is
        :math:`-\left[\partial_z\epsilon\partial_z+\rho_0'\right]\delta\phi=\rho + \partial_z\epsilon\partial_z\phi_0`.

        Args:
            activation: a factor (generally <=1) by which to multiply the determined
                change :math:`\delta\phi` before adding it.

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
        """ Simplest Fermi solver, `E_F=0` everywhere.

        Args:
            mesh: the :class:`~pynitride.mesh.Mesh` on which to perform the solve
        """
        self._mesh=mesh
        mesh.ensure_function_exists('EF',0)

    def solve(self):
        """ Sets `E_F=0` everywhere."""
        self._mesh['EF']=0

class Linear_Fermi():

    def __init__(self,mesh,contacts={'gate':0,'subs':-1}):
        """ Allows for a specified piecewise linear Fermi potential.

        An arbitrary number of "contacts" can be designated, and at each of these locations,
        a call to :func:`~pynitride.solvers.LinearFermi.solve` may specify a voltage.

        Args:
            mesh: the :class:`~pynitride.mesh.Mesh` on which to perform the solve
            contacts: a dictionary mapping names of contacts to locations in the mesh.
                Keys are arbitrary names, values are either (1) integers,
                in which case they will be interpreted as designating a layer interface
                (0 is the top surface, -1 is the bottom point), or (2) floats, in which
                case they will be interpreted as designating a nearest point to a `z`-value

        """
        self._mesh=mesh
        interfaces=[(0,None)]+mesh._interfacesp+[((len(mesh.zp)-1),None)]
        self._contacts=OrderedDict(sorted([(k,interfaces[v][0] if isinstance(v,int) else mesh.indexp(v))
                                   for k,v in contacts.items()],key=lambda x:x[1] if hasattr(x,'__getitem__') else x))
        mesh['EF']=NodFunction(mesh)

    def solve(self,**voltages):
        """ Sets the Fermi level to a linear interpolation of the specific values

        Args:
            **voltages: keyword arguments of the form `name=voltage` where `name` is
                one of the keys to the `contacts` dictionary supplied at initialization.

        """
        lefts=list(self._contacts.items())[:-1]
        rights=list(self._contacts.items())[1:]
        for (clname,cl),(crname,cr) in zip(lefts,rights):
            l=-voltages.get(clname,0)
            r=-voltages.get(crname,0)
            self._mesh['EF'][cl:(cr+1)]=(self._mesh.zp[cl:(cr+1)]-self._mesh.zp[cl])/(self._mesh.zp[cr]-self._mesh.zp[cl])*(r-l)+l


class SelfConsistentLoop():
    def __init__(self,fieldsolvers=[],carriermodels=[]):
        """ For Newton-iteration of field and carrier solvers.

        Dielectric ramping is provided for initial solutions and carrier models
        can be swapped in and out to allow for sequential solves by different
        methods.

        Args:
            fieldsolvers: a list of :class:`~pynitride.solvers.PoissonSolver`
            carriermodels: a list of :class:`~pynitride.carriers.CarrierModel`
        """
        self._fs=fieldsolvers
        self._cs=carriermodels

    def remove_carrier_model(self,cs):
        """ Remove a carrier solver from consideration.

        Args:
            cs: a :class:`~pynitride.carriers.CarrierModel` which should be in the
                list currently considered
        """
        self._cs.remove(cs)
    def add_carrier_model(self,cs):
        """ Add a carrier solver for consideration.

        Args:
            cs: a :class:`~pynitride.carriers.CarrierModel`
        """
        self._cs.append(cs)
    def swap_carrier_model(self,remove,add):
        """ Swaps out one :class:`~pynitride.carriers.CarrierModel` for another

        Args:
            remove: the model to remove
            add: the model to add
        """
        self.remove_carrier_model(remove)
        self.add_carrier_model(add)


    def newton_fields(self, activation=1):
        """ Perform one Newton step of the fields (just the fields, not the carriers).

        Calls the :func:`~pynitride.solvers.PoissonSolver.newton_step` for each field solver.

        Args:
            activation: the proportion by which to change the fields
                (see `~pynitride.solvers.PoissonSolver.newton_step`)
        Returns:
            the summed error from each field solver's `newton_step`

        """
        return sum(fs.newton_step(activation=activation) for fs in self._fs)

    def solve_fields(self):
        """ Perform a direct solve of the fields (just the fields, not the carriers).

        Calls the :func:`~pynitride.solvers.PoissonSolver.solve` for each field solver.

        """
        [fs.solve() for fs in self._fs]
    def solve_carriers(self):
        """ Perform a direct solve of the carriers (just the carriers, not the fields).

        Calls the :func:`~pynitride.carriers.CarrierModel.solve` for each.

        """
        [cs.solve_and_repopulate() for cs in self._cs]

    def loop(self, tol=1e-5, max_iter=100, min_activation=.05,
             init_activation=1,dec_activation=2,inc_activation=1.1):
        """ Loops the Newton field solution and carrier models until they agree

        If the error increases during a step, the step will be retried with a smaller `activation`
        (see :func:`~pynitride.solvers.PoissonSolver.newton_solve`).  If the `activation` becomes too small
        or the maximum number of iterations is passed, will raise an exception.

        Note, the first step in the solve is always the carriers, so they may be in any state before
        this function is called, whereas the fields must already be defined, eg by
        :func:`~pynitride.solvers.PoissonSolver.update_bands_to_potential`.

        Args:
            tol:  the absolute tolerance for the error
                as returned by :func:`~pynitride.SelfConsistentLoop.newton_step`
            max_iter: Maximum number of iterations allowed
                (not including any lower-activation re-attempts made)
            min_activation: the smallest activation allowed.
            init_activation: activation to start looping at
            dec_activation: factor by which to reduce the `activation` if a step fails
            inc_activation: factor by which to increase the `activation` if a step succeeds
        """
        with sublog("Starting SC loop"):

            # Start at infinite error and zero iterations
            err=np.inf
            i=0
            a=init_activation

            # Until consistency is reached
            while err>tol:

                # Limit iterations
                if i>=max_iter:
                    raise Exception("Maximum iteration reached in SC loop")

                # First solve carriers at fixed field
                self.solve_carriers()

                # Then step fields
                errprev=err
                err=self.newton_fields(activation=a)/a
                log("iter: {:3d}  err: {:.2e}  activ: {:g}".format(i,err,a))

                # If the error was not improved
                while err>errprev:

                    # Shrink the activation
                    a/=dec_activation
                    log("Retrying with Poisson activation={:g}".format(a))

                    # Limit activation shrinking
                    if a<min_activation:
                        raise Exception("Couldn't reduce error in SC loop")

                    # Reduce the change made by the previous field solve
                    for fs in self._fs:fs.shorten_last_step(1/dec_activation)

                    # Re-solve the carriers
                    self.solve_carriers()

                    # Try stepping the fields from this intermediate point
                    err=self.newton_fields(activation=a)/a
                    log("       iter: {:3d}  err: {:.2e}".format(i,err))

                # By now, a step was successful, so increase activation and iteration count
                a=min(inc_activation*a,1)
                i+=1
            log("Loop finished in {:2d} iterations with err={:g}".format(i,err))

    def ramp_epsfactor(self, start=1e4, stop=1, dlefstart=.1, dlefmax=.5,
                       dlefmin=.005,**loop_opts):
        """ Ramp the dielectric constant for an easy initial condition.

        Performs self-consistent loops at successive values of the `epsfactor`
        (see :func:`~pynitride.solvers.PoissonSolver.update_epsfactor`) from `start` until the `stop` value is
        reached. If the loops do not converge, smaller steps of the epsilon factor are attempted, this robustness is
        controlled by the `dlef` arguments, which constrain `dlef` (the logarithmic change in the epsilon factor from
        step to step, ie `log10(epsfactor)` changes by `dlef`).  If the step cannot be reduced further but the loop does
        not converge, an exception will be raised.


        Args:
            start: initial value of epsfactor
            stop: final value of epsfactor
            dlefstart: initial logarithmic delta for epsfactor stepping
            dlefmax: maximum allowed logarithmic delta for epsfactor stepping
            dlefmin: minimum allowed logarithmic delta for epsfactor stepping
            loop_opts: passed to each `pynitride.solvers.SelfConsistentLoop.loop`

        """
        with sublog("Starting eps factor ramp from {:g} to {:g}".format(start,stop)):

            # lef will be the log of the epsfactor
            lefstart=np.log10(start)
            lefstop=np.log10(stop)
            lef=lefstart

            # dlef will be the step of the lef
            dlefstart*=np.sign(lefstop-lefstart)
            dlef=dlefstart
            prevlef=None

            while True:
                ef=10**lef
                log("Eps factor: {:.2e}".format(ef))

                # Store the current fields
                for fs in self._fs:
                    fs.store_state()

                # Try solving at the new epsfactor
                try:

                    # Newton step the fields for the new epsfactor
                    fs.update_epsfactor(ef)
                    self.newton_fields()

                    # Do a SC loop
                    self.loop(**loop_opts)

                    # If that succeeded, we reach here, otherwise an exception was thrown

                    # If we're at the final epsfactor, we're done
                    if (lef-lefstop)<1e-9:
                        break

                    # Otherwise, go to the next epsfactor
                    else:
                        prevlef=lef

                        # Scale dlef since solve was successful
                        dlef=np.sign(dlef)*min(np.abs(2*dlef),np.abs(dlefmax))
                        lef=lef+dlef

                        # if we set dlef past the endpoint, go back
                        if np.sign(lef-lefstop)!=np.sign(lefstart-lefstop): lef=lefstop

                # If the solve failed at some epsfactor, try to recover
                except Exception as e:
                    log("Failure: {}".format(str(e)))

                    # If it's the first epsfactor, nothing we can do
                    if prevlef is None:
                        raise Exception("Failed at initial epsfactor")

                    # Otherwise, restore the previous consistent state
                    ef=10**prevlef
                    log("Restoring at {:.2e}".format(ef))
                    for fs in self._fs:
                        fs.restore_state()
                        fs.update_epsfactor(ef)
                    self.solve_carriers()

                    # Halve the step
                    dlef=dlef/2

                    # Limit step shrinking
                    if np.abs(dlef)<np.abs(dlefmin):
                        raise Exception("Eps factor step size too small")

                    # set the new epsfactor
                    lef=prevlef+dlef

                    # if we passed the end, go back
                    if np.sign(lef-lefstop)!=np.sign(lefstart-lefstop): lef=lefstop
            log("Done eps factor ramp")

