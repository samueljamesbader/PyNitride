import numpy as np
from pynitride.mesh import PointFunction, MidFunction, Function, ConstantFunction
from pynitride.paramdb import pmdb, k, pi, hbar, m_e, nm
from pynitride.cython_maths import fd12, fd12p
from pynitride.fem import assemble_stiffness_matrix, assemble_load_matrix, fem_eigsh
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigvalsh
from scipy.sparse import lil_matrix
from pynitride.visual import log, sublog
from operator import iadd,setitem
from pynitride.machine import Pool, glob_store_attributes, FakePool
from pynitride.reciprocal_mesh import KMesh2D
from operator import itemgetter

class CarrierModel():
    """ Superclass for all carrier models.

    A carrier model implements a :func:`solve()` function and a :func:`repopulate` function, which together define how
    to get from previously-solved bands to the induced carrier densities.
    """

    def __init__(self, mesh):
        """ Prep the mesh for any carrier model.
        """
        self.mesh=mesh

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
        """ Calls :func:`~solve` and :func:`~repopulate`"""
        return (self.solve(),self.repopulate())

class Semiclassical(CarrierModel):
    """ Implements a semiclassical carrier model.

    Carriers at each point are populated locally according to band alignment as if that point was the bulk.
    See :func:`solve` for details and see :ref:`Carrier Models <carriers_semiclassical>` for the physics.

    Arguments:
        mesh (:class:`~mesh.Mesh`): the Mesh to populate
        carriers (list of strings): which carriers to populate,
            eg ``["electron"]``, ``["hole"]``, or ``["electron","hole"]``
    """
    def __init__(self,mesh,carriers=['electron','hole']):
        super().__init__(mesh)
        m=self.mesh
        self._carriers=carriers
        if 'electron' in self._carriers:
            m.ensure_function_exists("n",value=0)
            m.ensure_function_exists("nderiv",value=0)
        if 'hole' in self._carriers:
            m.ensure_function_exists("p",value=0)
            m.ensure_function_exists("pderiv",value=0)

        # Compute the effective density of states
        if 'electron' in self._carriers:
            m.ensure_function_exists("Nc",dim=m.medos.shape[:-1],pos='mid')
            m['Nc']= m.eg * (m.medos*k*m.T/(2*pi*hbar**2))**(3/2)
        if 'hole' in self._carriers:
            m.ensure_function_exists("Nv",dim=m.mhdos.shape[:-1],pos='mid')
            m['Nv']= m.hg * (m.mhdos*k*m.T/(2*pi*hbar**2))**(3/2)

    def solve(self):
        """ Does nothing.

        For Semiclassical, since there are no expensive eigenvalue calculations, all the work is done in
        :func:`~repopulate`.
        """
        pass

    def repopulate(self,Ec_eff=None, Ev_eff=None, addon=False):
        """ Populate carriers semi-classically into the mesh.  See :func:`CarrierModel.solve` for conditions,
        and see :ref:`Carrier Models <carriers_semiclassical>` for the physics.

        Accepts effective conduction and valence bands as optional arguments to use in place of the mesh `Ec` and `Ev`.
        For use as a stand-alone model, these should not be supplied, but they are convenient when this model is called
        by other models which may populate some levels quantum mechanically and then use this function to fill in the
        rest of the band semiclassically.  In this case, the addon option, which adds the computed density to whatever
        is on the mesh, rather than replacing it, may also come in handy.

        Arguments:
            Ec_eff (Node :class:`.mesh.Function`): Effective conduction band level, optional.

            Ev_eff (Node :class:`.mesh.Function`): Effective valence band level, optional.

            addon (bool): Whether to add (True) the computed density to current mesh values or simply replace the
            current values (False).  Default False.
        """
        m=self.mesh

        # Use effective band edges if provided, otherwise take from mesh
        Ec_eff = Ec_eff if (Ec_eff is not None) else m.Ec
        Ev_eff = Ev_eff if (Ev_eff is not None) else m.Ev

        # A function which either adds the calculation to the mesh value or replaces it depending on addon
        assign= (lambda x,y: iadd(x,y)) if addon else (lambda x,y: setitem(x,slice(None),y))

        # Compute the carrier density and its derivative
        if 'electron' in self._carriers:
            eta=((m.EF.tmf() - Ec_eff) - m.cDE) / (k * m.T)
            assign(m['n'],     np.sum(m.Nc * fd12(eta), axis=0).tpf())
            assign(m['nderiv'],np.sum(-(m.Nc/(k*m.T)) * fd12p(eta), axis=0).tpf())
        if 'hole' in self._carriers:
            eta=((Ev_eff - m.EF.tmf()) - m.vDE) / (k * m.T)
            assign(m['p'],     np.sum(m.Nv * fd12(eta), axis=0).tpf())
            assign(m['pderiv'],np.sum((m.Nv/(k*m.T)) * fd12p(eta), axis=0).tpf())


class Schrodinger(CarrierModel):
    """ Implements a Schrodinger envelope function carrier model.

    Carriers at each point are populated according to band alignment as if that point was the bulk.
    See :func:`solve` for details and see :ref:`Carrier Models <carriers>` for the physics.

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
            m.ensure_function_exists("n",value=0)
            m.ensure_function_exists("nderiv",value=0)
            if self._blend: self._sce=Semiclassical(m,carriers=['electron'])
        if 'hole' in self._carriers:
            m.ensure_function_exists("p",value=0)
            m.ensure_function_exists("pderiv",value=0)
            if self._blend: self._sch=Semiclassical(m,carriers=['hole'])

        self._subbands=[]
        self._zkinetic=[]
        self._lkineticfactor=[]
        if 'electron' in self._carriers:
            self._nebands=m.mez.shape[0]
            self._een=np.empty([self._nebands, self._neig+self._blend])
            self._epsi=PointFunction(m, empty=(self._nebands, self._neig+self._blend),dtype=float)
            self._subbands+=[{'carrier':'electron','subband':l} for l in range(self._nebands)]
        if 'hole' in self._carriers:
            self._nhbands=m.mhz.shape[0]
            self._hen=np.empty([self._nhbands, self._neig+self._blend])
            self._hpsi=PointFunction(m, empty=(self._nhbands, self._neig+self._blend),dtype=float)
            self._subbands+=[{'carrier':'hole','subband':l} for l in range(self._nhbands)]

        for sb in self._subbands:
            elec,l=(sb['carrier']=='electron'),sb['subband']
            b= -1 if elec else 1
            mz=(m.mez if elec else m.mhz)[l,:]
            mxy=(m.mexy if elec else m.mhxy)[l,:]
            kperp=0
            sb['C0_kinetic']=np.expand_dims(np.expand_dims(hbar**2/(2*mxy)*kperp**2,0),0)
            sb['C2']=np.expand_dims(np.expand_dims(hbar**2/(2*mz),0),0)
            sb['energies']=self._een if elec else self._hen
            sb['psi']=self._epsi if elec else self._hpsi

        self._M=assemble_load_matrix(w=MidFunction(m,1),dzp=m._dzp,
                              n=1,dirichelet1=True,dirichelet2=True)

    def solve(self):
        m=self.mesh

        for sb in self._subbands:
            elec,l=(sb['carrier']=='electron'), sb['subband']
            U= (m.Ec+m.cDE[l,:]) if elec else -(m.Ev-m.vDE[l,:])
            O=MidFunction(m,[[0.]])
            H=assemble_stiffness_matrix(
                C0=U+sb['C0_kinetic'],
                Cl=None,Cr=None,
                C2=sb['C2'],
                dzp=m._dzp,
                dirichelet1=True,dirichelet2=True
            )
            fem_eigsh(stiffness_matrix=H,load_matrix=self._M,
                      eigval_out=sb['energies'][l],eigvec_out=sb['psi'][l,:,:],n=1,
                      dirichelet1=True,dirichelet2=True,
                      k=self._neig+self._blend,sigma=float(np.min(U)))

            if not elec:
                sb['energies'][l]*=-1


    def repopulate(self):
        m=self.mesh

        for f in ['n','p','nderiv','pderiv']: m.ensure_function_exists(f)
        if 'electron' in self._carriers:
            m['n']=m['nderiv']=0
        if 'hole' in self._carriers:
            m['p']=m['pderiv']=0

        for sb in self._subbands:
            elec,l=(sb['carrier']=='electron'), sb['subband']
            b,mxy,g,dens,deriv= (-1,m.mexy[l],m.eg[l],m.n,m.nderiv) if elec else (1,m.mhxy[l],m.hg[l],m.p,m.pderiv)
            en=np.expand_dims(sb['energies'][l,:],2)
            psi=sb['psi'][l,:,:]

            eta=b*(en-m.EF).tmf()/(k*m.T)
            psisq=abs(psi.tmf())**2
            mmean=np.atleast_2d((mxy*psisq).integrate(definite=True)).T
            dens+= np.sum(  (g*mmean*k*m.T)/(2*pi*hbar**2)*psisq*np.log(1+np.exp(eta)),      axis=0).tpf()
            deriv+=np.sum(b*(g*mmean)      /(2*pi*hbar**2)*psisq*np.exp(eta)/(1+np.exp(eta)),axis=0).tpf()

        if self._blend:
            if 'electron' in self._carriers:
                self._sce.repopulate(Ec_eff=np.maximum(np.expand_dims(self._een[:,-1],1),m.Ec),addon=True)
            if 'hole' in self._carriers:
                self._sch.repopulate(Ev_eff=np.minimum(np.expand_dims(self._hen[:,-1],1),m.Ev),addon=True)

# TODO: Blending for multiband kp
@glob_store_attributes('mesh','rmesh','_Cmats','_load_matrix')
class MultibandKP(CarrierModel):
    def __init__(self,mesh,rmesh=None,num_eigenvalues=20):
        """ Solves the multiband k.p problem for the valence bands.

        Can use either a 1D line of k-points or a rectangular 'xy' grid of k-points or a polar grid of k-points
        depending on `kmeshmethod`.  Regardless, assumes in-plane inversion symmetry (ie for 1D, only solves at
        positive kx, and for 2D only solves one quadrant of k-plane for efficiency.

        :param mesh: the :py:class:`pynitride.mesh.Mesh` on which to solve
        :param num_eigenvalues: (int) the number of eigenvalues to solve for at each k-point
        """
        super().__init__(mesh)
        m=mesh
        assert len(m._matblocks)==1, "kp only works on a mesh with a single material system for now"
        self._neig=num_eigenvalues
        self.rmesh=rmesh

        if rmesh is not None:
            m.ensure_function_exists("p",value=0)
            m.ensure_function_exists("pderiv",value=0)

            self._Cmats=m._matblocks[0].matsys.kp_Cmats(m, kx=self.rmesh.kx, ky=self.rmesh.ky)

            # Initialize other functions
            self.rmesh['kppsi']=PointFunction(m,dtype='complex',empty=(self.rmesh.N,num_eigenvalues,6,))
            self.rmesh['kpen']=np.empty((self.rmesh.N,self._neig))
            self.rmesh['normsqs']=PointFunction(m,dtype='float',empty=(self.rmesh.N,num_eigenvalues))
        self._load_matrix=assemble_load_matrix(m.ones_mid,m.dzp,n=6,dirichelet1=True,dirichelet2=True)


    # kpen is kt, eig, z
    # kppsi is kt, eig, comp, z
    # normsqs is kt, eig, z
    def solve(self):
        log("MBKP Solve",level="debug")
        m=self.mesh
        kpen,kppsi,normsqs=itemgetter('kpen','kppsi','normsqs')(self.rmesh)
        Pool.process_pool(new=True)
        def save_one_solve(ik):
            kpen[ik,:],kppsi[ik,:,:,:],normsqs[ik,:,:]= \
                Pool.process_pool().apply(self.solve_one_k,args=(None,None,ik))
        Pool.thread_pool().map(save_one_solve,range(self.rmesh.N))

    # kpen is eig, z
    # kppsi is eig, comp, z
    # normsqs is eig, z
    def solve_one_k(self,kx,ky,ik=None):
        m=self.mesh
        if ik is not None:
            C0_kin,Cl,Cr,C2=self._Cmats[ik]
        else:
            C0_kin,Cl,Cr,C2=m._matblocks[0].matsys.kp_Cmats(m,kx=[kx],ky=[ky])[0]

        # TODO: pot can be a property of Multiband rather than re-forming for each k
        pot=(m.Ev+m.EvOffset)
        C0=C0_kin+np.expand_dims(np.eye(C0_kin.shape[0]),2)*pot
        H=-assemble_stiffness_matrix(C0,Cl,Cr,C2,m.dzp,dirichelet1=True,dirichelet2=True)
        eigvals=np.empty([self._neig])
        eigvecs=PointFunction(m,np.empty([self._neig,6,m.Np],dtype=complex),dtype=complex)
        # Use pairwise GS to re-orthogonalize, since Laczos is bad at orthogonalizing degenerate eigenvectors
        fem_eigsh(H,self._load_matrix,eigvals,eigvecs,6,dirichelet1=True,dirichelet2=True,pairwise_GS=True,
                  k=self._neig,sigma=float(np.min(-pot)),which='LM',tol=0,ncv=self._neig*2)

        # eig, z
        normsqs=np.sum(abs(eigvecs)**2,axis=1)
        return -eigvals,eigvecs,normsqs

    def solve_point_as_bulk(self,zp):
        m=self.mesh
        kt=self._kt
        izp=m.indexp(zp)

        return np.array([eigvalsh(C[0][:,:,izp]) for i, (kti,C) in enumerate(zip(kt,self._Cmats))])


    def repopulate(self):
        m=self.mesh
        kT=k*m.T
        # kt, eig, z
        eta=(np.expand_dims(self.rmesh['kpen'],2)-m.EF)/kT.tpf()

        p=(1/(4*pi**2))*self.rmesh.integrate(self.rmesh['normsqs']/(1+np.exp(-eta)))
        m['p']=(1/(4*pi**2))*self.rmesh.integrate(
            np.sum(self.rmesh['normsqs']/(1+np.exp(-eta)),axis=1))
        m['pderiv']=(1/(4*pi**2))*self.rmesh.integrate(
            np.sum(self.rmesh['normsqs']*(np.exp(-eta))/(1+np.exp(-eta))**2/kT.tpf(),axis=1))

        log("not blending",level="TODO")



