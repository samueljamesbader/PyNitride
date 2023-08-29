from functools import partial
from operator import iadd, setitem

import numpy as np
from numpy.linalg import eigh
from pynitride.core.cython_maths import fd12, fd12p
from pynitride.core.fem import assemble_stiffness_matrix, assemble_load_matrix, fem_eigsh
from pynitride import NodFunction, MidFunction
from pynitride import kb, pi, hbar
from pynitride import log

from pynitride.core.machine import process_pool, glob_store_attributes, raiser


class CarrierModel():

    def __init__(self, mesh):
        """ Superclass for all carrier models.

        A carrier model implements a :func:`solve()` function and a :func:`repopulate` function, which together define how
        to get from previously-solved bands to the induced carrier densities.
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
            m['Nc']= m.eg * (m.medos * kb * m.T / (2 * pi * hbar ** 2)) ** (3 / 2)
        if 'hole' in self._carriers:
            m.ensure_function_exists("Nv",dim=m.mhdos.shape[:-1],pos='mid')
            m['Nv']= m.hg * (m.mhdos * kb * m.T / (2 * pi * hbar ** 2)) ** (3 / 2)

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
            eta=((m.EF.tmf() - Ec_eff) - m.cDE) / (kb * m.T)
            assign(m['n'],     np.sum(m.Nc * fd12(eta), axis=0).tpf())
            assign(m['nderiv'], np.sum(-(m.Nc / (kb * m.T)) * fd12p(eta), axis=0).tpf())
        if 'hole' in self._carriers:
            eta=((Ev_eff - m.EF.tmf()) - m.vDE) / (kb * m.T)
            assign(m['p'],     np.sum(m.Nv * fd12(eta), axis=0).tpf())
            assign(m['pderiv'], np.sum((m.Nv / (kb * m.T)) * fd12p(eta), axis=0).tpf())


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
            assert 'schro_e_psi' not in m, "This mesh already has an electron Schrodinger"
            m.designate_non_mesh_item('schro_e_en')

            self._nebands=m.mez.shape[0]
            m['schro_e_en']=self._een=np.empty([self._nebands, self._neig+self._blend])
            m['schro_e_psi']=self._epsi=NodFunction(m, empty=(self._nebands, self._neig+self._blend),dtype=float)
            self._subbands+=[{'carrier':'electron','subband':l} for l in range(self._nebands)]

        if 'hole' in self._carriers:
            assert 'schro_h_psi' not in m, "This mesh already has an hole Schrodinger"
            m.designate_non_mesh_item('schro_h_en')

            self._nhbands=m.mhz.shape[0]
            m['schro_h_en']=self._hen=np.empty([self._nhbands, self._neig+self._blend])
            m['schro_h_psi']=self._hpsi=NodFunction(m, empty=(self._nhbands, self._neig+self._blend),dtype=float)
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

        self._M=assemble_load_matrix(w=MidFunction(m,1),dzn=m._dzn,
                              n=1,dirichelet1=True,dirichelet2=True)

    def solve(self):
        """ Solves the Schrodinger eigenvalue problem, call before :func:`Schrodinger.repopulate`"""
        m=self.mesh

        for sb in self._subbands:
            elec,l=(sb['carrier']=='electron'), sb['subband']
            U= (m.Ec+m.cDE[l,:]) if elec else -(m.Ev-m.vDE[l,:])
            O=MidFunction(m,[[0.]])
            H=assemble_stiffness_matrix(
                C0=U+sb['C0_kinetic'],
                Cl=None,Cr=None,
                C2=sb['C2'],
                dzn=m._dzn,
                dirichelet1=True,dirichelet2=True
            )
            fem_eigsh(stiffness_matrix=H,load_matrix=self._M,
                      eigval_out=sb['energies'][l],eigvec_out=sb['psi'][l,:,:],n=1,
                      dirichelet1=True,dirichelet2=True,
                      k=self._neig+self._blend,sigma=float(np.min(U)))

            if not elec:
                sb['energies'][l]*=-1


    def repopulate(self):
        """ Populates the carrier densities onto the mesh, call after :func:`Schrodinger.solve`"""
        m=self.mesh

        for f in ['n','p','nderiv','pderiv']: m.ensure_function_exists(f)
        if 'electron' in self._carriers:
            m['n']=m['nderiv']=0
        if 'hole' in self._carriers:
            m['p']=m['pderiv']=0

        for sb in self._subbands:
            elec,l=(sb['carrier']=='electron'), sb['subband']
            b,mxy,g,dens,deriv= (-1,m.mexy[l],m.eg[l],m.n,m.nderiv) if elec else (1,m.mhxy[l],m.hg[l],m.p,m.pderiv)
            en=np.expand_dims(sb['energies'][l,:],1)
            psi=sb['psi'][l,:,:]

            eta=b*(en-m.EF).tmf()/(kb * m.T)
            psisq=abs(psi.tmf())**2
            mmean=np.atleast_2d((mxy*psisq).integrate(definite=True)).T
            #import pdb; pdb.set_trace()
            dens+= np.sum((g * mmean * kb * m.T) / (2 * pi * hbar ** 2) * psisq * np.log(1 + np.exp(eta)), axis=0).tpf()
            deriv+=np.sum(b*(g*mmean)      /(2*pi*hbar**2)*psisq*np.exp(eta)/(1+np.exp(eta)),axis=0).tpf()

        if self._blend:
            if 'electron' in self._carriers:
                self._sce.repopulate(Ec_eff=np.maximum(np.expand_dims(self._een[:,-1],1),m.Ec),addon=True)
            if 'hole' in self._carriers:
                self._sch.repopulate(Ev_eff=np.minimum(np.expand_dims(self._hen[:,-1],1),m.Ev),addon=True)

# TODO: Blending for multiband kp
@glob_store_attributes('mesh','rmesh','_Cmats','_load_matrix','_enbv')
class MultibandKP(CarrierModel):
    def __init__(self,mesh,rmesh=None,num_eigenvalues=20,carriers=['hole']):
        """ Solves the multiband k.p problem for the valence bands.

        Can use either a 1D line of k-points or a rectangular 'xy' grid of k-points or a polar grid of k-points
        depending on `kmeshmethod`.  Regardless, assumes in-plane inversion symmetry (ie for 1D, only solves at
        positive kx, and for 2D only solves one quadrant of k-plane for efficiency.

        Args:
            mesh: the :py:class:`pynitride.mesh.Mesh` on which to solve
            rmesh: the reciprocal mesh on which to solve (if you only want to use
                :func:`MultibandKP.solve_one_k`, this is not needed)
            num_eigenvalues: (int) the number of eigenvalues to solve for at each k-point
            carriers: `['hole']` or `['electron']`, can't do both yet.
        """
        super().__init__(mesh)
        m=mesh
        assert len(carriers)==1, "kp only works with one type of carrier at a time right now"
        self._carrier=carriers[0]

        assert len(m._matblocks)==1, "kp only works on a mesh with a single material system for now"
        self._n=m._matblocks[0].matsys.kp_dim[self._carrier]
        self._neig=num_eigenvalues
        self.rmesh=rmesh


        self._interp_ready=False

        if rmesh is not None:
            self._Cmats=m._matblocks[0].matsys.kp_Cmats(m, kx=self.rmesh.kx, ky=self.rmesh.ky, carrier=self._carrier)
            if np.array(m.zm).shape==(): return

            if self._carrier=='electron':
                m.ensure_function_exists("n",value=0)
                m.ensure_function_exists("nderiv",value=0)
            if self._carrier=='hole':
                m.ensure_function_exists("p",value=0)
                m.ensure_function_exists("pderiv",value=0)

            # Initialize other functions
            if 'kpen' not in self.rmesh:
                self.rmesh['kpen']=np.empty((self.rmesh.N,self._neig))
            if 'kppsi' not in self.rmesh:
                self.rmesh['kppsi']=NodFunction(m,dtype=self._Cmats[0][0].dtype,empty=(self.rmesh.N,num_eigenvalues,self._n,))
            if 'normsqs' not in self.rmesh:
                self.rmesh['normsqs']=NodFunction(m,dtype='float',empty=(self.rmesh.N,num_eigenvalues))
        self._load_matrix=assemble_load_matrix(m.ones_mid, m.dzn, n=self._n, dirichelet1=True, dirichelet2=True)


    @property
    def kppsi(self):
        """ The spinor wavefunctions, shape `(len(kt), num_eigs, matsys.kp_dim, mesh.Nn)`."""
        return self.rmesh['kppsi']
    @property
    def kpen(self):
        """ The energies, shape `(len(kt), num_eigs, mesh.Nn)`."""
        return self.rmesh['kpen']
    @property
    def normsqs(self):
        """ The norm-squareds of the wavefunction, shape `(len(kt), num_eigs, mesh.Nn)`."""
        return self.rmesh['normsqs']


    # kpen is kt, eig, z
    # kppsi is kt, eig, comp, z
    # normsqs is kt, eig, z
    def solve(self):
        """ Solves the MBKP eigenvalue problem, call before :func:`MultibandKP.repopulate`"""
        self._interp_ready=False

        def save_solve(ik,res):
            self.kpen[ik,:],self.kppsi[ik,:,:,:],self.normsqs[ik,:,:]= res
        pool=process_pool(new=True)
        asyncs=[pool.apply_async(self.solve_one_k,args=(None,None,ik),
                 callback=partial(save_solve,ik),error_callback=raiser)
            for ik in range(self.rmesh.N)]
        for asyn in asyncs: asyn.wait()


    def solve_one_k(self,kx,ky,ik=None):
        """ Solves the k.p problem for just one wavevector.

        Args:
            kx,ky: the in-plane wavevector at which to compute the k.p matrices to solve
            ik: if specified, this index into the RMesh will be used to find the wavevector instead of the kx,ky
            arguments.  This also allows the pre-calculated k.p matrices to be used.

        Returns:
            a tuple of
            eigenvalues (shape `num_eigs`),
            eigenvectors (shape `num_eigs, matsys.kp_dim, mesh.Nn`), and
            normsqs (shape `num_eigs, mesh.Nn`)
        """
        m=self.mesh
        if ik is not None:
            C0_kin,Cl,Cr,C2=self._Cmats[ik]
        else:
            C0_kin,Cl,Cr,C2=m._matblocks[0].matsys.kp_Cmats(m,kx=[kx],ky=[ky],carrier=self._carrier)[0]

        # Note: pot can be a property of Multiband rather than re-forming for each k
        if self._carrier=='hole':
            pot=(m.Ev+m.EvOffset)
            ascending=False
            sigma=float(np.max(pot))
        if self._carrier=='electron':
            pot=(m.Ec+m.EcOffset)
            ascending=True
            sigma=float(np.min(pot))
        C0=C0_kin+np.expand_dims(np.eye(C0_kin.shape[0]),2)*pot
        H=assemble_stiffness_matrix(C0, Cl, Cr, C2, m.dzn, dirichelet1=True, dirichelet2=True)
        eigvals=np.empty([self._neig])
        eigvecs=NodFunction(m,np.empty([self._neig,self._n,m.Nn],dtype=H.dtype),dtype=H.dtype)
        # Use pairwise GS to re-orthogonalize, since Lanczos is bad at orthogonalizing degenerate eigenvectors
        fem_eigsh(H,self._load_matrix,eigvals,eigvecs,self._n,dirichelet1=True,dirichelet2=True,pairwise_GS=True,
            ascending=ascending,k=self._neig,sigma=sigma,which='LM',tol=0,ncv=self._neig*2)

        # eig, z
        normsqs=np.sum(abs(eigvecs)**2,axis=1)
        return eigvals,eigvecs,normsqs

    def solve_point_as_bulk(self,zn=None,kz=0):
        """ Solves the transverse dispersion at a given point as if the material properties there were in bulk.

        Args:
            zn: for an actual mesh, `zn` specifies a z-coord to solve at.  Or if a :func:`Material.bulk` is used instead
                to create the `MultibandKP`, then `zn=None` is fine.
            kz: z-wavevector to solve at

        """
        m=self.mesh
        assert self._carrier=='hole'
        if zn is not None:
            izn=m.indexn(zn)
            solved=[eigh(-(C[0][:,:,izn]+2*kz*C[1][:,:,izn]+kz**2*C[3][:,:,izn])) for i, C in enumerate(self._Cmats)]
            return np.array([-s[0]+m.EvOffset[izn] for s in solved]), np.array([s[1].T for s in solved])
        else:
            solved=[eigh(-(C[0][:,:]+2*kz*C[1][:,:]+kz**2*C[3][:,:])) for i, C in enumerate(self._Cmats)]
            return np.array([-s[0]+m.EvOffset for s in solved]), np.array([s[1].T for s in solved])



    def repopulate(self):
        """ Populates the carrier densities onto the mesh, call after :func:`MultibandKP.solve`"""
        m=self.mesh
        kT= kb * m.T

        c,cderiv,sign=['n','nderiv',-1] if self._carrier=='electron' else ['p','pderiv',+1]
        eta=sign*(m.EF-np.expand_dims(self.rmesh['kpen'],2))/kT.tpf()
        m[c]=(1/(4*pi**2))*self.rmesh.integrate(
            np.sum(self.rmesh['normsqs']/(1+np.exp(eta)),axis=1))
        m[cderiv]=(1/(4*pi**2))*self.rmesh.integrate(
            sign*np.sum(self.rmesh['normsqs']*(np.exp(eta))/(1+np.exp(eta))**2/kT.tpf(),axis=1))

        log("not blending",level="TODO")

    def _get_interpolation(self):
        if not self._interp_ready:
            self._enbv=[self.rmesh.interpolator(self.rmesh['kpen'][:,eig])
                        for eig in range(self._neig)]
            self._interp_ready=True

    def interp_energy(self,absk,theta,eig,bounds_check=True,grid=False):
        """ Returns spline-interpolated results for the energy

        Args:
            absk: radial coordinate
            theta: angular coordinate
            eig: which eigenvalue
            bounds_check: whether to throw out-of-bounds errors
            grid: as in :py:class:`scipy.interpolate.BivariateSpline`

        Returns:
            return shape determined by grid as in :py:class:`scipy.interpolate.BivariateSpline`

        """
        self._get_interpolation()
        return self._enbv[eig](absk,theta,bounds_check=bounds_check,grid=grid)

    def interp_radial_group_velocity(self,absk,theta,eig,bounds_check=True,grid=False):
        """ Returns spline-interpolated results for the group velocity in the radial direction

        Args:
            absk: radial coordinate
            theta: angular coordinate
            eig: which eigenvalue
            bounds_check: whether to throw out-of-bounds errors
            grid: as in :py:class:`scipy.interpolate.BivariateSpline`

        Returns:
            return shape determined by grid as in :py:class:`scipy.interpolate.BivariateSpline`

        """
        self._get_interpolation()
        return 1/hbar*self._enbv[eig](absk,theta,dabsk=1,bounds_check=bounds_check,grid=grid)

    def interp_group_velocity(self,absk,theta,eig,bounds_check=True,grid=False):
        """ Returns spline-interpolated results for the 2D-vector group velocity

        Args:
            absk: radial coordinate
            theta: angular coordinate
            eig: which eigenvalue
            bounds_check: whether to throw out-of-bounds errors
            grid: as in :py:class:`scipy.interpolate.BivariateSpline`

        Returns:
            tuple of `(vx, vy)`, return shape of each determined by grid
            as in :py:class:`scipy.interpolate.BivariateSpline`

        """
        self._get_interpolation()
        v_r=1/hbar*self._enbv[eig](absk,theta,dabsk=1 ,bounds_check=bounds_check,grid=grid)
        v_t=1/hbar*self._enbv[eig](absk,theta,dtheta=1,bounds_check=bounds_check,grid=grid)/self.rmesh.absk
        v_x=v_r*np.cos(theta)-v_t*np.sin(theta)
        v_y=v_r*np.sin(theta)+v_t*np.cos(theta)
        return v_x,v_y

    def interp_radial_eff_mass(self,absk,theta,eig,bounds_check=True):
        """ Returns spline-interpolated results for the effective mass in the radial direction

        ie if evaluated at theta=0, then this is inversely proportional to the second x-derivative of energy.
        A minus sign is applied to hole effective masses to make them generically positive.

        Args:
            absk: radial coordinate
            theta: angular coordinate
            eig: which eigenvalue
            bounds_check: whether to throw out-of-bounds errors
            grid: as in :py:class:`scipy.interpolate.BivariateSpline`

        Returns:
            return shape determined by grid as in :py:class:`scipy.interpolate.BivariateSpline`

        """
        self._get_interpolation()
        sign= 1 if self._carrier=='electron' else -1
        return sign/(1/hbar**2*self._enbv[eig](absk,theta,dabsk=2,bounds_check=bounds_check))

