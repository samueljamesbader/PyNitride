import numbers

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from pynitride.poissolve.maths import tdma, fd12, fd12p
from pynitride import ParamDB, MaterialFunction, PointFunction, ConstantFunction, MidFunction, SubMesh

pmdb=ParamDB(units='neu')
k,hbar,q=pmdb.quantity("k,hbar,e")


class SchrodingerSolver():
    def __init__(self,mesh,carriers=['electron','hole']):
        r""" Solves the Schrodinger equation along *z* for the lowest eigenstates in a potential well.

        :param mesh: the :py:class:`~poissolve.mesh.structure.Mesh` on which the Schrodinger problem is defined
        :param carriers: list of carriers (elements may be 'electron' or 'hole') to quantize
       """
        self._mesh=m=mesh
        self._dopants=FermiDirac3D.identifydopants(mesh)
        self._Nc,self._Nv=FermiDirac3D.effective_dos_3d(mesh)
        self._cDE,self._vDE=FermiDirac3D.band_edge_shifts(mesh)

        self._props={c:{} for c in carriers}
        for carrier,v in self._props.items():
            bands=mesh._layers[0].material[carrier,"band"]
            for i,b in enumerate(bands):
                v[b]={}
                v[b]['T']=self.z_kinetic_term(m,MaterialFunction(m,[carrier,b,'mzs']))
                v[b]['mxys']=MaterialFunction(m,[carrier,b,'mxys'],pos='point')
                v[b]['g']=mesh._layers[0].material[[carrier,b,'g']] # can't vary spatially
                v[b]['DE']=MaterialFunction(m,[carrier,b,'DE'],pos='point')
            m[{'electron':'Ec_eff','hole':'Ev_eff'}[carrier]]=PointFunction(m,empty=(len(bands),))

        for k in ['n','p','nderiv','pderiv']:
            if k not in m:
                m[k]=PointFunction(m,empty=())

    @staticmethod
    def z_kinetic_term(mesh,mz=None):
        r""" Generates the symmetrized *z* kinetic energy term for use in a Schrodinger solution.

        This tridiagonal matrix is the discrete, symmetrized representation of
        :math:`\frac{-\hbar^2}{2m_z}\frac{\partial^2\psi}{\partial z^2}`.
        See :ref:`Solving the Schrodinger Equation <solve_schrodinger__1d>` for details on the discrete form.


        :param mesh: the :py:class:`~poissolve.mesh.structure.Mesh` on which the Schrodinger problem is defined
        :param mz: MidFunction of the effective mass along *z*
        :return: the *z* kinetic term as a sparse (CSC) matrix
        """
        diagonal=(hbar**2/(mz*mesh._dz)).to_point_function(interp='unweighted')/mesh._dzp
        offdiagonal=-(hbar**2/(2*mz*mesh._dz *np.sqrt(mesh._dzp[:-1]*mesh._dzp[1:])))
        T=diags([offdiagonal,diagonal,offdiagonal],[-1,0,1],format='csc')
        return T

    @staticmethod
    def lateral_kinetic_term(mesh,kperp,mxy):
        r""" Generates the lateral kinetic term for use in a Schrodinger solution.

        This diagonal matrix is the discrete version  of :math:`\frac{\hbar^2k_\perp^2}{2m}`.
        See :ref:`Solving the Schrodinger Equation <solve_schrodinger>` for details on the discrete form.

        :param mesh: the Mesh on which the Schrodinger problem is defined
        :param kperp: norm of the lateral wavevector
        :param mxy: MidFunction of the lateral effective mass
        :return: the lateral kinetic term as a sparse (CSC) matrix
        """
        diagonal=(hbar**2*kperp**2/(2*mxy)).to_point_function(interp='unweighted')
        T=diags(diagonal,format='csc')
        return T

    @staticmethod
    def solve_schrodinger_problem(mesh,z_kinetic_term,potential,lateral_kinetic_term=0,
                                  num_eigenvalues=8,psi_out=None):
        if not psi_out: psi_out=PointFunction(mesh,empty=(num_eigenvalues,))

        H=z_kinetic_term+diags(potential)+lateral_kinetic_term
        energies,eigenvectors=eigsh(H,k=num_eigenvalues,sigma=np.min(potential))
        psi_out[:,:]=(1/np.sqrt(mesh._dzp))*eigenvectors.T

        return energies, psi_out

    @staticmethod
    def carrier_density(psi,g,mxys,eta,kT,summed=True):
        return (g/(2*np.pi)*kT/hbar**2)* \
              np.sum(mxys*(psi**2*(np.log(1+np.exp(eta)))).T,axis=1)


    # (kT)**-1 * d(carrier_density)/d(eta)
    @staticmethod
    def carrier_density_deriv(psi,g,mxys,eta):
        return (-g/(2*np.pi)/hbar**2)* \
               np.sum(mxys*(psi**2*(1+np.exp(-eta))**-1).T,axis=1)

    def solve(self,activation=1, eff_mass_average=True):
        m=self._mesh
        EF=m['EF']
        kT=k*m.pmdb['T']

        for carrier,bands in self._props.items():
            electron,hole=(carrier=="electron"),(carrier=="hole")
            conc=0
            deriv=0
            for i,(b, bandparms) in enumerate(bands.items()):
                abbrev="_"+carrier[0]+"_"+b

                if electron: U=(m['Ec']+bandparms['DE'])
                elif hole:   U=-(m['Ev']-bandparms['DE'])

                E,Psi=self.solve_schrodinger_problem(m,bandparms['T'],U)
                if hole: E=-E

                m['Psi'+abbrev]=Psi
                m['Energies'+abbrev]=E_i=ConstantFunction(m,E)

                assert eff_mass_average, "Solving with full k-integral is not supported."
                meff=1/((Psi**2/bandparms['mxys']).integrate()[:,-1])

                if electron: eta=(m['EF']-E_i)/kT
                elif hole:   eta=(E_i-m['EF'])/kT
                conc+=self.carrier_density(Psi,bandparms['g'],meff,eta,kT)
                deriv+=self.carrier_density_deriv(Psi,bandparms['g'],meff,eta)

                if electron: np.maximum(E_i[-1],m['Ec'],out=m['Ec_eff'][i,:])
                elif hole:   np.minimum(E_i[-1],m['Ev'],out=m['Ev_eff'][i,:])
            if hole: deriv=-deriv
            m[{'electron':'n','hole':'p'}[carrier]]=conc
            m[{'electron':'nderiv','hole':'pderiv'}[carrier]]=deriv
        carriers=self._props.keys()
        if 'hole' not in carriers:
            m['p']=0
            m['pderiv']=0
        if 'electron' not in carriers:
            m['n']=0
            m['nderiv']=0
        for key,v in zip(['n','p','nderiv','pderiv'],
                FermiDirac3D.carrier_density(EF,
                    Ec=m['Ec_eff'] if 'electron' in carriers else m['Ec'],
                    Ev=m['Ev_eff'] if 'hole' in carriers else m['Ev'],
                    Nc=self._Nc,Nv=self._Nv,kT=kT,
                    conduction_band_shifts=0 if 'electron' in carriers else self._cDE,
                    valence_band_shifts=0 if 'hole' in carriers else self._vDE,
                    compute_derivs=True)):
            m[key]+=v

        m['Ndp'],m['Nam'],m['Ndpderiv'],m['Namderiv']= \
            FermiDirac3D.ionized_donor_density(m,EF,m['Ec'],m['Ev'],kT,self._dopants,compute_derivs=True)

        if activation!=1:
            for key in ['n','p','nderiv','pderiv','Ndp','Nam','Ndpderiv','Namderiv']:
                m[key]*=activation

        m['rho']=activation*m['rho_pol']+q*(m['p']+m['Ndp']-m['n']-m['Nam'])
        m['rhoderiv']= q*(m['pderiv']+m['Ndpderiv']-m['nderiv']-m['Namderiv'])


class PoissonSolver():
    r""" Solves the Poisson equation on a mesh.

    The boundary conditions assumed are that the potential is zero at the first mesh point, and the electric field
    goes to zero at the last mesh point (or, to be more precise, at the next midpoint after the last meshpoint).
    Two solve functions are available: :py:func:`~pynitride.poissolve.solvers_old.poisson.solve` and
    :py:func:`~pynitride.poissolve.solvers_old.poisson.PoissonSolver.isolve`.  The former is a full, direct solution, which can be obtained
    directly from charge integration.  The latter is a Newton-method solver appropriate for self-consistent
    iteration with a charge solver such as FermiDirac3D or Schrodinger.

    :param mesh: the :py:class:`~pynitride.poissolve.mesh.structure.Mesh` on which to perform the solve
    """
    def __init__(self, mesh):
        self._mesh = mesh

        if isinstance(mesh._layers.surface,numbers.Real):
            self._phib=mesh._layers.surface
        else:
            self._phib=mesh._layers[0].material('surface={}.electronbarrier'.format(mesh._layers.surface))

        # ARE THESE NECESSARY
        eps=self._eps=mesh['eps']= MaterialFunction(mesh, 'dielectric.eps')
        mesh['mqV']= PointFunction(mesh, 0.0)
        mesh['DEc']= MaterialFunction(mesh, 'electron.DEc', pos='point')
        self._Eg= MaterialFunction(mesh,'Eg', pos='point')


        self._left=np.empty(len(mesh._z))
        self._right=np.empty(len(mesh._z))
        self._left[1:]=eps/(mesh._dz * mesh._dzp[1:])
        self._right[:-1]=eps/(mesh._dz * mesh._dzp[:-1])
        self._center=-MidFunction(mesh,eps/mesh._dz).to_point_function(interp='unweighted')/mesh._dzp
        #self._center=np.zeros(len(mesh.z))
        #self._center[1:-1]=-1/(mesh._dz[1:]*mesh._dz[:-1])
        self._center[-1]=self._center[-2]

        self._left[:2]=0
        self._right[0]=0
        self._right[-1:]=0
        self._center[0]=1
        self._center[1:-1]*=2

        self._mqV_temp = PointFunction(mesh)  # temp

    def solve(self):
        m=self._mesh
        qrho=q*m['rho']
        qrho[0]=0
        m['mqV']=tdma(self._left,self._center,self._right,qrho)
        self._update_others()
        mqV=m['mqV']
        m['E']=mqV.differentiate()
        m['D']=MidFunction(m,self._eps*m['E'])
        m['arho2']=m['D'].differentiate()

    def _update_others(self):
        m=self._mesh
        m['Ec']=m['mqV']+m['EF'][0]+self._phib+m['DEc']
        m['Ev']=m['Ec']-self._Eg

    def isolve(self,visual=False):
        m=self._mesh
        qrho=q*m['rho']
        qrho[0]=0

        # left right and center are for +d^2/dx^2, ie center is negative
        # isolve uses -d^2/dx^2, ie center (without rhoderiv) is positive

        diag = -self._center + q * self._mesh['rhoderiv']
        diag[0] -= q * m['rhoderiv'][0]
        diag[-1] -= q * m['rhoderiv'][-1]

        a=-self._left
        b=diag
        c=-self._right

        d= (q*m['arho2'] - qrho)
        d[0]=0
        d[-1]=-m['D'][-1]/m._dzp[-1]

        # What I had after redoing Neumann at bottom
        d[-1]=-m['rho'][-1]-m['D'][-1]/m._dzp[-1]

        ## Trying to just fix the last point of D at zero
        d[-1]=-m['D'][-1]/m._dzp[-1]



        import numpy as np
        if visual:
            import matplotlib.pyplot as mpl
            mpl.figure()
            mpl.subplot(311)
            mpl.plot(m.z,qrho-q*self._rhoprev)
            print(np.max(np.abs(qrho-q*self._rhoprev)))
            mpl.title('rho- rhoprev')
            mpl.subplot(312)
            mpl.plot(m.z,m['rhoderiv'])
            mpl.title('rhoderiv')
            mpl.tight_layout()

        dqmV=tdma(a,b,c,d)
        #print(dqmV[-15:])
        m['mqV']+=dqmV
        self._update_others()
        self._rhoprev=m['rho'].copy()


        mqV=m['mqV']
        m['E']=mqV.differentiate()
        m['D']=MidFunction(m,self._eps*m['E'])
        m['arho2']=m['D'].differentiate()


        return np.max(np.abs(dqmV))#np.sum(np.abs(dqmV))/np.sum(np.abs(m['mqV']))


class FermiDirac3D():
    def __init__(self,mesh):
        self._mesh=mesh

        self._Nc,self._Nv=self.effective_dos_3d(mesh)
        self._cDE,self._vDE=self.band_edge_shifts(mesh)
        self._dopants=self.identifydopants(mesh)


    @staticmethod
    def identifydopants(mesh):
        dopants={'Donor':{},'Acceptor':{}}
        for d in [k[:-10] for k in mesh if k.endswith("ActiveConc")]:

            types=set(t for t in (l.material.get('dopant='+d+'.type',default=None) for l in mesh._layers) if t is not None)
            if len(types)>1: raise Exception(
                "Can't have one dopant be acceptor in one material and donor in another.  "\
                "You'll have to use two separate dopant names.  Sorry. ")
            if len(types)==1:
                dopants[list(types)[0]][d]={'conc':mesh[d+'ActiveConc']}
            else:
                print("No materials include {} as a dopant.".format(d))
        for doptype in dopants.keys():
            for d,v in dopants[doptype].items():
                v['E']=MaterialFunction(mesh,d+'.E',pos='point')
                v['g']=MaterialFunction(mesh,d+'.g',pos='point')
        mesh['Nd']=np.sum(d['conc'] for d in dopants['Donor'].values())\
            if len(dopants['Donor']) else ConstantFunction(mesh,0)
        mesh['Na']=np.sum(d['conc'] for d in dopants['Acceptor'].values()) \
            if len(dopants['Acceptor']) else ConstantFunction(mesh,0)
        return dopants

    @staticmethod
    def effective_dos_3d(mesh):
        kT=k*mesh.pmdb['T']
        # We'll have to confirm these formulae (factors of 2?) later...
        Nc=MaterialFunction(mesh,pos='point',prop=lambda mat:
            mat['electron.band.g']*(mat['electron.band.mdos']*kT/(2*np.pi*hbar**2))**(3/2))
        Nv=MaterialFunction(mesh,pos='point',prop=lambda mat:
            mat['hole.band.g']*(mat['hole.band.mdos']*kT/(2*np.pi*hbar**2))**(3/2))
        return Nc,Nv

    @staticmethod
    def band_edge_shifts(mesh):
        cDE=MaterialFunction(mesh,pos='point',prop=lambda mat: mat['electron.band.DE'])
        vDE=MaterialFunction(mesh,pos='point',prop=lambda mat: mat['hole.band.DE'])
        return cDE,vDE

    @staticmethod
    def carrier_density(EF,Ec,Ev,Nc,Nv,kT,conduction_band_shifts=None,valence_band_shifts=None,compute_derivs=True):

        # Can save a copy and loop by checking if shifts are zero
        Ec_eff=Ec if conduction_band_shifts is None else Ec+conduction_band_shifts
        Ev_eff=Ev if valence_band_shifts is None else Ev-valence_band_shifts

        # Can save time by moving the summation inside the FD integral?
        n=np.sum(Nc*fd12((EF-Ec_eff)/kT),axis=0)
        p=np.sum(Nv*fd12((Ev_eff-EF)/kT),axis=0)

        if compute_derivs:
            nderiv=np.sum(-(Nc/kT)*fd12p((EF-Ec_eff)/kT),axis=0)
            pderiv=np.sum((Nv/kT)*fd12p((Ev_eff-EF)/kT),axis=0)
            return n,p,nderiv,pderiv
        else:
            return n,p

    @staticmethod
    def ionized_donor_density(mesh,EF,Ec,Ev,kT,dopants,compute_derivs=True):

        # Tiwari Compound Semiconductor Devices pg31-32

        Ndp=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']*(1/(1+d['g']*np.exp((EF-Ec+d["E"])/kT)))
           for d in dopants['Donor'].values())))
        Nam=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']*(1/(1+d['g']*np.exp((Ev+d["E"]-EF)/kT)))
           for d in dopants['Acceptor'].values())))

        if compute_derivs:
            Ndpderiv=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']*\
                    (d['g']/kT)*np.exp((EF-Ec+d["E"])/kT)/(1+d['g']*np.exp((EF-Ec+d["E"])/kT))**2
                for d in dopants['Donor'].values())))
            Namderiv=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']*\
                    (-d['g']/kT)*np.exp((Ev+d["E"]-EF)/kT)/(1+d['g']*np.exp((Ev+d["E"]-EF)/kT))**2
                for d in dopants['Acceptor'].values())))
            return Ndp,Nam,Ndpderiv,Namderiv
        else:
            return Ndp,Nam

    def solve(self,activation=1, quantum_band_shift=False):
        m=self._mesh
        EF=m['EF']
        Ec=m['Ec']
        Ev=m['Ev']
        kT=k*m.pmdb['T']

        m['Ndp'],m['Nam'],m['Ndpderiv'],m['Namderiv']=\
            self.ionized_donor_density(m,EF,Ec,Ev,kT,self._dopants,compute_derivs=True)

        if quantum_band_shift:
            m['n'],m['p'],m['nderiv'],m['pderiv']=self.carrier_density(EF,Ec,Ev,self._Nc,self._Nv,kT,
                conduction_band_shifts=self._cDE,valence_band_shifts=self._vDE,compute_derivs=True)
        else:
            m['n'],m['p'],m['nderiv'],m['pderiv']=self.carrier_density(EF,Ec,Ev,self._Nc,self._Nv,kT,
               conduction_band_shifts=None,valence_band_shifts=None,compute_derivs=True)

        if activation!=1:
            for key in ['n','p','nderiv','pderiv','Ndp','Nam','Ndpderiv','Namderiv']:
                m[key]*=activation

        m['rho']=activation*m['rho_pol']+q*(m['p']+m['Ndp']-m['n']-m['Nam'])
        m['rhoderiv']= q*(m['pderiv']+m['Ndpderiv']-m['nderiv']-m['Namderiv'])


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

        # Prep solvers_old
        self._ps=PoissonSolver(m)
        self._fd=FermiDirac3D(m)


    def solve(self, low_act=4, rise=500, tol=1e-8, max_iter=100, callback=lambda *args: None):
        self._ps.solve()
        if callback(): return
        for activation in np.logspace(-low_act,-0.,rise):
            self._fd.solve(activation=activation)
            err=self._ps.isolve(visual=False)
            print(err)
            if callback(): return
        print("Rose")
        for i in range(max_iter):
            self._fd.solve(activation=1)
            err=self._ps.isolve(visual=False)
            if callback(): return
            print(err)
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

        # Prep solvers_old
        self._ps=PoissonSolver(m)





    def solve(self, low_act=4, rise=500, tol=1e-10, max_iter=100, callback=lambda *args: None):
        self._ps.solve()
        #if callback(): return
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
                print("Semi-classical success (max err={:.2g}) after {:d} refinement iterations".format(err,i-1))
                break
        assert err<tol, "Stopped because reached max_iter with err ({:.2g}) > tol ({:.2g}).".format(err,tol)
        for i in range(max_iter):
            for s in self._quantum_charge_solvers:
                if isinstance(s,SchrodingerSolver):
                    s.solve(activation=1)
                elif isinstance(s,FermiDirac3D):
                    s.solve(activation=1,quantum_band_shift=True)
            err=self._ps.isolve(visual=False)
            if callback(): return
            if err<tol:
                print("Full success (max err={:.2g}) after {:d} refinement iterations".format(err,i-1))
                break
        assert err<tol, "Stopped because reached max_iter with err ({:.2g}) > tol ({:.2g}).".format(err,tol)