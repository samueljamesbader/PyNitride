import numpy as np
from pynitride.mesh import PointFunction, MidFunction, Function, ConstantFunction
from pynitride.paramdb import pmdb, k, pi, hbar, m_e, nm
from pynitride.maths import fd12, fd12p, assemble6x6
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigvalsh
from scipy.sparse import lil_matrix
from pynitride.visual import log, sublog
from operator import iadd,setitem

class CarrierModel():
    """ Superclass for all carrier models.

    A carrier model implements a :func:`solve()` function and a :func:`repopulate` function, which together define how
    to get from previously-solved bands to the induced carrier densities.
    """

    def __init__(self, mesh):
        """ Prep the mesh for any carrier model.
        """
        self._mesh=mesh

    def solve(self):
        """ Performs any prep work needed (e.g. solving for wavefunctions) before the mesh can be populated.

        This method generally does the expensive eigenvalue calculations, whereas :func:`repopulate` simply takes these
        results and fills in the levels.

        Precondition: the mesh already includes the bands "Ec" and "Ev" (e.g. provided by :class:`Poisson`).

        Post-condition: One can now call repopulate()
        """
        raise NotImplementedError

    def repopulate(self):
        """ Populates the mesh with carriers.

        This method should be quick because :func:`solve` has already done the heavy lifting.
        So this can be called inside tighter loops where the Fermi level changes but bands do not.

        Precondition: self.solve() has already been called to set up any necessary resources (eg wavefunctions)
        and the mesh includes the Fermi level "EF" (e.g. provided by :class:`Equilibrium`).

        Post-condition: the mesh will have the electron and hole densities "n" and "p",
        as well as their approximate functional derivatives "nderiv" and "pderiv" with respect to the bands
        (ie :math:`n' = \delta n / \delta [-q \phi],\ p' = \delta p / \delta [-q \phi]`)
        """
        raise NotImplementedError

    def solve_and_repopulate(self):
        return (self.solve(),self.repopulate())

class Semiclassical(CarrierModel):
    """ Implements a semiclassical carrier model.

    Carriers at each point are populated locally according to band alignment as if that point was the bulk.
    See :func:`solve` for details and see :ref:`Carrier Models <carriers_semiclassical>` for the physics.

    Arguments:
        mesh (:class:`.mesh.Mesh`): the Mesh to populate

        carriers (list of strings): which carriers to populate,
        eg ``["electron"]``, ``["hole"]``, or ``["electron","hole"]``
    """
    def __init__(self,mesh,carriers=['electron','hole']):
        super().__init__(mesh)
        m=self._mesh
        self._carriers=carriers
        m.request_function("EF")
        if 'electron' in self._carriers:
            m.request_functions(["eg","medos","Ec"])
            m.ensure_function_exists("n",value=0)
            m.ensure_function_exists("nderiv",value=0)
        if 'hole' in self._carriers:
            m.request_functions(["hg","mhdos","Ev"])
            m.ensure_function_exists("p",value=0)
            m.ensure_function_exists("pderiv",value=0)


    def initialize(self):
        m=self._mesh
        # Compute the effective density of states
        if 'electron' in self._carriers:
            m['Nc']= m.eg * (m.medos*k*m.T/(2*pi*hbar**2))**(3/2)
        if 'hole' in self._carriers:
            m['Nv']= m.hg * (m.mhdos*k*m.T/(2*pi*hbar**2))**(3/2)

    def solve(self):
        """ Does nothing."""
        pass

    def repopulate(self,Ec_eff=None, Ev_eff=None, addon=False):
        """ Populate carriers semi-classically into the mesh.  See :func:`CarrierModel.solve` for conditions,
        and see :ref:`Carrier Models <carriers_semiclassical>` for the physics.

        Accepts effective condition and valence bands as optional arguments to use in place of the mesh Ec and Ev.
        For use as a stand-alone model, these should not be supplied, but they are convenient when this model is called
        by other models which may populate some levels quantum mechanically and then use this function to fill in the
        rest of the band semiclassically.  In this case, the addon option, which adds the computed density to whatever
        is on the mesh, rather than replacing it, may also come in handy.

        Arguments:
            Ec_eff (Point :class:`.mesh.Function`): Effective conduction band level, optional.

            Ev_eff (Point :class:`.mesh.Function`): Effective valence band level, optional.

            addon (bool): Whether to add (True) the computed density to current mesh values or simply replace the
            current values (False).  Default False.
        """
        m=self._mesh

        # Use effective band edges if provided, otherwise take from mesh
        Ec_eff = Ec_eff if (Ec_eff is not None) else m.Ec
        Ev_eff = Ev_eff if (Ev_eff is not None) else m.Ev

        assign= (lambda x,y: iadd(x,y)) if addon else (lambda x,y: setitem(x,slice(None),y))

        # Compute the carrier density and its derivative
        if 'electron' in self._carriers:
            eta=((m.EF - Ec_eff).tmf() - m.cDE) / (k * m.T)
            assign(m['n'],     np.sum(m.Nc * fd12(eta), axis=0).tpf())
            assign(m['nderiv'],np.sum(-(m.Nc/(k*m.T)) * fd12p(eta), axis=0).tpf())
        if 'hole' in self._carriers:
            eta=((Ev_eff - m.EF).tmf() - m.vDE) / (k * m.T)
            assign(m['p'],     np.sum(m.Nv * fd12(eta), axis=0).tpf())
            assign(m['pderiv'],np.sum((m.Nv/(k*m.T)) * fd12p(eta), axis=0).tpf())

class Schrodinger(CarrierModel):
    """ Implements a Schrodinger envelope function carrier model.

    Carriers at each point are populated according to band alignment as if that point was the bulk.
    See :func:`solve` for details and see :ref:`Carrier Models <carriers_schrodinger>` for the physics.

    Arguments:
        mesh (:class:`.mesh.Mesh`): the Mesh to populate

        carriers (list of strings): which carriers to populate, eg ``["electron"]``, ``["hole"]``, or ``["electron","hole"]``

        num_eigenvalues (int): how many eigenvalues to solve for and occupy quantum mechanically. [Default = 8]

        blend (bool): whether to add in semiclassically-occupied carriers at energies beyond the solved for levels. [Default = True]

        transverse (str): "parabolic" [Default] or "full k-space", see physics section for details
    """
    def __init__(self,mesh,carriers=['electron','hole'],
                 num_eigenvalues=8,blend=True,transverse="parabolic",
                 boundary=["Dirichlet","Dirichlet"]):
        super().__init__(mesh)
        m=mesh

        # Store values
        self._blend=blend
        if transverse != "parabolic":
            raise NotImplementedError
        self._transverse="parabolic"
        self._carriers=carriers
        self._neig=num_eigenvalues
        self._boundary=boundary
        self._blend=blend

        if 'electron' in self._carriers:
            m.request_functions(["mez","mexy","Ec","cDE"])
            m.ensure_function_exists("n",value=0)
            m.ensure_function_exists("nderiv",value=0)
            if self._blend: self._sce=Semiclassical(m,carriers=['electron'])
        if 'hole' in self._carriers:
            m.request_functions(["mhz","mhxy","Ev","vDE"])
            m.ensure_function_exists("p",value=0)
            m.ensure_function_exists("pderiv",value=0)
            if self._blend: self._sch=Semiclassical(m,carriers=['hole'])

    def initialize(self):
        m=self._mesh
        self._subbands=[]
        self._zkinetic=[]
        self._lkineticfactor=[]
        if 'electron' in self._carriers:
            self._nebands=m.mez.shape[0]
            m['een']=PointFunction(m, empty=(self._nebands, self._neig))
            m['epsi']=PointFunction(m, empty=(self._nebands, self._neig))
            self._subbands+=[{'carrier':'electron','subband':l} for l in range(self._nebands)]
            if self._blend: self._sce.initialize()
        if 'hole' in self._carriers:
            self._nhbands=m.mhz.shape[0]
            m['hen']=PointFunction(m, empty=(self._nhbands, self._neig))
            m['hpsi']=PointFunction(m, empty=(self._nhbands, self._neig))
            self._subbands+=[{'carrier':'hole','subband':l} for l in range(self._nhbands)]
            if self._blend: self._sch.initialize()

        for sb in self._subbands:
            elec,l=(sb['carrier']=='electron'),sb['subband']
            b= -1 if elec else 1
            mz=(m.mez if elec else m.mhz)[l,:]
            mxy=(m.mexy if elec else m.mhxy)[l,:]
            diagonal=-b*(hbar**2/(mz*m._dzp)).tpf(interp='unweighted')/m._dzm
            offdiagonal=b*(hbar**2/(2*mz*m._dzp *np.sqrt(m._dzm[:-1]*m._dzm[1:])))
            sb['zkinetic']=\
                diags([offdiagonal,diagonal,offdiagonal],[-1,0,1],format='csc')
            sb['lkinetic']=\
                diags(-b*(hbar**2/(2*mxy)).tpf(interp='unweighted'),format='csc')
            sb['energies']=m['een'] if elec else m['hen']
            sb['psi']=m['epsi'] if elec else m['hpsi']

    def solve(self):
        kperp=0
        m=self._mesh

        for sb in self._subbands:
            elec,l=(sb['carrier']=='electron'), sb['subband']
            U= (m.Ec+m.cDE[l,:].tpf()) if elec else (m.Ev-m.vDE[l,:].tpf())
            valley=np.min(U) if elec else np.max(U)
            H=sb['zkinetic']+diags(U)+sb['lkinetic']*kperp**2
            #sb['H']=H
            #sb['valley']=valley
            #sb['U']=U
            #print("l: ",l," elec: ",elec)
            #print('eigs0',eigsh(-H,k=self._neig+self._blend,sigma=-valley)[0])
            if elec:
                energies,eigenvectors=eigsh(H, k=self._neig+self._blend,sigma= valley)
            else:
                energies,eigenvectors=eigsh(-H,k=self._neig+self._blend,sigma=-valley)
                energies=-energies
                #print("EN: ", energies)
                #print("V: ",valley)
                #print("H:\n",H[:10,:10].todense())
                #print(eigsh(-H,k=self._neig+self._blend,sigma=-valley)[0])
            sb['energies'][l,:,:]=np.atleast_2d(energies[:self._neig]).T
            sb['psi'][l,:,:]=(1/np.sqrt(m._dzm))*eigenvectors[:,:self._neig].T


    def repopulate(self):
        m=self._mesh

        for f in ['n','p','nderiv','pderiv']: m.ensure_function_exists(f)
        if 'electron' in self._carriers:
            m['n']=m['nderiv']=0
        if 'hole' in self._carriers:
            m['p']=m['pderiv']=0

        for sb in self._subbands:
            elec,l=(sb['carrier']=='electron'), sb['subband']
            b,mxy,g,dens,deriv= (-1,m.mexy[l],m.eg[l],m.n,m.nderiv) if elec else (1,m.mhxy[l],m.hg[l],m.p,m.pderiv)
            en=sb['energies'][l,:,:]
            psi=sb['psi'][l,:,:]

            eta=b*(en-m.EF).tmf()/(k*m.T)
            psisq=abs(psi.tmf())**2
            mmean=np.atleast_2d((mxy*psisq).integrate(definite=True)).T
            #print("dens ",dens[10])
            dens+= np.sum(  (g*mmean*k*m.T)/(2*pi*hbar**2)*psisq*np.log(1+np.exp(eta)),      axis=0).tpf()
            deriv+=np.sum(b*(g*mmean)      /(2*pi*hbar**2)*psisq*np.exp(eta)/(1+np.exp(eta)),axis=0).tpf()
            #print("dens ",dens[10])

        if self._blend:
            if 'electron' in self._carriers:
                self._sce.repopulate(Ec_eff=np.maximum(m.een[:,-1,:],m.Ec),addon=True)
            if 'hole' in self._carriers:
                self._sch.repopulate(Ev_eff=np.minimum(m.hen[:,-1,:],m.Ev),addon=True)

class MultibandKP(CarrierModel):
    def __init__(self,mesh,num_eigenvalues=20,ktmax=2/nm,num_kpoints=25, kmeshmethod='1D'):
        """ Solves the multiband k.p problem for the valence bands.

        Can use either a 1D line of k-points or a rectangular 'xy' grid of k-points or a polar grid of k-points
        depending on `kmeshmethod`.  Regardless, assumes in-plane inversion symmetry (ie for 1D, only solves at
        positive kx, and for 2D only solves one quadrant of k-plane for efficiency.

        :param mesh: the :py:class:`pynitride.mesh.Mesh` on which to solve
        :param num_eigenvalues: (int) the number of eigenvalues to solve for at each k-point
        :param ktmax: the max absolute k value to solve at, either (1) an int if just solving along just along one
        direction in k-space or (2) a two-tuple of ints (for x and y directions) if solving the full in-plane k-space
        with `kmeshmethod='xy'`.
        :param num_kpoints: The number of points to solve for per direction, either (1) an int if just solving one
        direction or (2) a two-tuple of ints if solving the full in-plane dispersion.  In the latter case, the
        interpretation depends on `kmeshmethod`.
        in-plane dispersion without spherical symmetry, the number of points goes as square of this value.)
        :param kmeshmethod: Either '1D' or 'xy' or 'polar'. If (1) just solving in one-direction, specify '1D' and
        supply single integers for ktmax and num_kpoints. If (2) solving the full-plane and 'xy' is specified, the two
        integers of num_kpoints are taken to the be the number of k-points in the x and y directions.
        If 'polar' is specified, the first num_kpoints value is the number of radial points and the second value is
        the numer of angles along which to solve (within one quadrant).
        """
        super().__init__(mesh)
        m=mesh
        assert len(m._matblocks)==1, "kp only works on a mesh with a single material system for now"
        m.ensure_function_exists("p",value=0)
        m.ensure_function_exists("pderiv",value=0)
        self._neig=num_eigenvalues
        self._ktmax=ktmax
        self._num_kpoints=num_kpoints
        self._kmeshmethod=kmeshmethod

    def initialize(self):
        m=self._mesh
        ktmax=self._ktmax
        num_kpoints=self._num_kpoints
        num_eigenvalues=self._neig
        kmeshmethod=self._kmeshmethod

        # Make the k-mesh
        if isinstance(self._kmeshmethod,str):
            if kmeshmethod=='1D':
                self._kt=np.linspace(0,ktmax,num_kpoints)
                self._kx=1*self._kt
                self._ky=0*self._kt
            elif kmeshmethod=='xy':
                if not hasattr(ktmax,'__iter__'): ktmax=[ktmax,ktmax]
                if not hasattr(num_kpoints,'__iter__'): num_kpoints=[num_kpoints,num_kpoints]
                kx=np.linspace(0,ktmax[0],num_kpoints[0])
                ky=np.linspace(0,ktmax[1],num_kpoints[1])
                self._kmeshman=KMesh2D(kx,ky)
                self._kt=self._kmeshman.kt
                self._kx=self._kmeshman.kx1
                self._ky=self._kmeshman.ky1
            elif kmeshmethod=='xyfull':
                if not hasattr(ktmax,'__iter__'): ktmax=[ktmax,ktmax]
                if not hasattr(num_kpoints,'__iter__'): num_kpoints=[num_kpoints,num_kpoints]
                kx=np.linspace(-ktmax[0],ktmax[0],num_kpoints[0])
                ky=np.linspace(-ktmax[1],ktmax[1],num_kpoints[1])
                self._kmeshman=KMesh2D(kx,ky)
                self._kt=self._kmeshman.kt
                self._kx=self._kmeshman.kx1
                self._ky=self._kmeshman.ky1
            elif kmeshmethod=='polar':
                raise NotImplementedError
            kx,ky=self._kmeshman.kx,self._kmeshman.ky
        # otherwise kmeshmethod should be an iterable of ((kx1,ky1),(kx2,ky2),...)
        else :
            self._kt=np.array(kmeshmethod)
            self._kx,self._ky=np.hsplit(self._kt,2)
            kx,ky=self._kx,self._ky

        # Assemble k.p matrices
        log("Assembling k.p matrices ...",level='info')
        self._Cmats=Cmats=m._matblocks[0].matsys.kp_Cmats(m,kx=kx,ky=ky)
        self._H=[assemble6x6(C0,Cl,Cr,C2,m._dzm,m._dzp,periodic=False) for [C0,Cl,Cr,C2] in Cmats]
        log("Done assembly.",level='info')

        # Initialize other functions
        m['kppsi']=PointFunction(m,dtype='complex',empty=(len(self._kt),num_eigenvalues,6,))
        m['kpen']=PointFunction(m,dtype='float',empty=(len(self._kt),num_eigenvalues,))
        self._normsqs=PointFunction(m,dtype='float',empty=(len(self._kt),num_eigenvalues))

    def remesh(self,num_eigenvalues=20,ktmax=2/nm,num_kpoints=25, kmeshmethod='1D'):
        if 'kppsi' in self._mesh._functions:
            del self._mesh._functions['kppsi']
        if 'kpen' in self._mesh._functions:
            del self._mesh._functions['kpen']
        self.__init__(self._mesh,num_eigenvalues,ktmax,num_kpoints,kmeshmethod)
        self.initialize()

    def solve(self):
        log("MBKP Solve",level="debug")
        m=self._mesh
        kt=self._kt
        pot=-np.reshape(np.reshape(np.tile(m.Ev+m.EvOffset.tpf(),6),(6,len(m.zp))).transpose(),(6*len(m.zp)))
        #m.tests=[]
        for i,(kti,H) in enumerate(zip(kt,self._H)):
            Htot=-H+diags(pot)
            energies,eigenvectors=eigsh(Htot,k=self._neig,sigma=np.min(pot),which='LM',tol=0,ncv=self._neig*2)

            # Sort by energy
            indarr=np.argsort(energies)
            # kt, eig, z
            m['kpen'][i,:,:]=-np.atleast_2d(energies[indarr]).T
            # kt, eig, comp, z
            m['kppsi'][i,:,:,:]=np.swapaxes(np.reshape(
                    eigenvectors[:,indarr],
                    (len(m._zp),6,self._neig)),0,2)\
                /np.sqrt(m._dzm)
            # kt, eig, z
            self._normsqs[i,:,:]=np.sum(abs(m.kppsi[i])**2,axis=1)

            # delete this
            #m.tests+=[[-Htot,eigenvectors[:,indarr],-energies[indarr]]]

    def solve_point_as_bulk(self,zp):
        m=self._mesh
        kt=self._kt
        izp=m.indexp(zp)

        return np.array([eigvalsh(C[0][:,:,izp]) for i, (kti,C) in enumerate(zip(kt,self._Cmats))])


    def repopulate(self):
        m=self._mesh
        kT=k*m.T
        kt=self._kt
        # kt, eig, z
        eta=(m.kpen-m.EF)/kT.tpf()

        if self._kmeshmethod=='1D':
            m['p']=np.sum(1/(2*np.pi)*np.trapz(kt*(self._normsqs/(1+np.exp(-eta))).T,x=kt),axis=1)
            m['pderiv']=np.sum(1/(2*np.pi*kT.tpf())*np.trapz(kt*(self._normsqs*(np.exp(-eta))/(1+np.exp(-eta))**2).T,x=kt).T,axis=0)
        elif self._kmeshmethod=='xy':
            dkx,dky=self._kmeshman.dkx,self._kmeshman.dky
            ig= self._normsqs/(1+np.exp(-eta))
            igd=self._normsqs*(np.exp(-eta))/(1+np.exp(-eta))**2

            # x 4 for all quadrants
            m['p']=4*np.sum(1/(2*np.pi)**2*self._kmeshman.intflat(ig),axis=0)
            m['pderiv']=4*np.sum(1/(4*np.pi**2*kT.tpf())*self._kmeshman.intflat(igd),axis=0)
        elif self._kmeshmethod=='xyfull':
            dkx,dky=self._kmeshman.dkx,self._kmeshman.dky
            ig= self._normsqs/(1+np.exp(-eta))
            igd=self._normsqs*(np.exp(-eta))/(1+np.exp(-eta))**2

            m['p']=np.sum(1/(2*np.pi)**2*self._kmeshman.intflat(ig),axis=0)
            m['pderiv']=np.sum(1/(4*np.pi**2*kT.tpf())*self._kmeshman.intflat(igd),axis=0)
        log("not blending",level="TODO")


class KMesh2D:
    def __init__(self,kx,ky):
        self.kx1=kx
        self.kx1p=kx[kx>=0]
        self.ky1=ky
        self.ky1p=ky[ky>=0]
        self.lkx=len(self.kx1)
        self.lky=len(self.ky1)
        self.KX,self.KY=np.meshgrid(kx,ky)
        self.KT=np.stack([self.KX,self.KY],axis=2)
        self.kt=self.conv2flat(self.KT)
        self.kx=self.kt[:,0]
        self.ky=self.kt[:,1]
        self.dkx=kx[1]-kx[0]
        self.dky=ky[1]-ky[0]

    def conv2grid(self,arr):
        return np.reshape(arr,[self.lky,self.lkx]+list(arr.shape[1:]))
    def conv2flat(self,arr):
        return np.reshape(arr,[np.prod(arr.shape[:2])]+list(arr.shape[2:]))
    def intflat(self,arr):
        ig= self.conv2grid(arr).T
        return np.trapz(np.trapz(ig,dx=self.dky),dx=self.dkx).T
    def along(self,arr,dir='x',input='guess',onesided=True):
        assert input=='flat' or (input=='guess' and arr.shape[0]==self.lkx*self.lky)
        if dir=='x':
            iy=np.argmax(self.ky1==0)
            assert (self.ky1[iy]==0), "0 not in ky"
            ix = np.argmax(self.kx1>=0) if onesided else 0
            return arr[(iy*self.lkx+ix):(iy*self.lkx+self.lkx)]
        elif dir=='y':
            ix=np.argmax(self.kx1==0)
            assert (self.kx1[ix]==0), "0 not in kx"
            iy=np.argmax(self.ky1>=0) if onesided else 0
            return arr[(ix+iy*self.lkx)::self.lkx]
