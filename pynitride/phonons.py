from pynitride.machine import Pool, glob_store_attributes, FakePool, Counter, raiser
from pynitride.visual import log, sublog
from pynitride.mesh import PointFunction
from pynitride.paramdb import hbar, meV
from pynitride.fem import assemble_stiffness_matrix, assemble_load_matrix, fem_eigsh, fem_solve
from pynitride.maths import polar2cart
from pynitride.material import AlGaN
from scipy.sparse.linalg import eigsh
from functools import partial
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from itertools import product
from collections import OrderedDict
import numpy as np
pi=np.pi

class PhononModel():

    def __init__(self, mesh, rmesh, keepmesh=None):
        """ Superclass for all phonon models."""
        self._mesh=mesh
        self._keepmesh=mesh if keepmesh is None else keepmesh
        self.rmesh=rmesh
        self._interp_ready=False

    def solve(self):
        pass

    @property
    def q(self): return self.rmesh.absk1
    @property
    def en(self): return self.rmesh['en']
    @property
    def vecs(self): return self.rmesh['vecs']

    def save(self,filename,just_energies=False):
        keys= ['en'] if just_energies else ['en','vecs']
        self.rmesh.save(filename,keys=keys)
    def read(self,name,just_energies=False):
        if just_energies:
            self.rmesh.read(name,keys='en')
        else:
            self.rmesh.read(name)
        if 'en' in self.rmesh:
            print("En shape ",self.en.shape)
            assert self.en.shape==(self.rmesh.N,self._neig),\
                "Loaded PhononModel does not match current"
        if 'vecs' in self.rmesh:
            print("Vecs shape ",self.vecs.shape)
            print("Not checking vecs")
            assert self.vecs.shape==(self.rmesh.N,self._neig,self._n,self._keepmesh.Np),\
                "Loaded PhononModel does not match current"
            self.rmesh['vecs']=PointFunction(self._keepmesh,self.vecs,dtype=self.vecs.dtype)


    def _get_interpolation(self):
        if not self._interp_ready:
            self._splines=[self.rmesh.interpolator(self.rmesh['en'][:,eig])
                        for eig in range(self._neig)]
            self._interp_ready=True

    def interp_energy(self,absk,eig,bounds_check=True):
        self._get_interpolation()
        return self._splines[eig](absk,bounds_check=bounds_check)

    def interp_radial_group_velocity(self,absk,eig,bounds_check=True):
        self._get_interpolation()
        return 1/hbar*self._splines[eig](absk,dabsk=1,bounds_check=bounds_check)

    def I2(self,carrier,psii,psij,iq,thetaq,l):
        """ The matrix element squared between two wavefunctions

        Args:
            psii, psij: the two wavefunctions (as 2-D PointFunctions)
            iq (int): index into the `q` array
            thetaq: in-plane angle of the phonon propagation (from X toward Y)
            l: which eigenvalue to use of those solved for

        Returns:
            the squared matrix element
        """
        pass

class AcousticPhonon(PhononModel):
    
    def __init__(self,mesh,rmesh,keepmesh=None,
            vecform='XYZ',deformation=True,piezo=False):
        r""" Superclass for acoustic phonons.

        Args:
            mesh,rmesh,keepmesh: see :class:`~PhononModel`
            vecform (str): Format for the vecs, 'XZ', 'XYZ', or 'Y'
            deformation (bool): whether to include deformation potential
                effects in matrix elements
            piezo (bool): whether to solve for the piezoelectric potential
                induced by the phonon and use it in matrix elements

        """
        super().__init__(mesh,rmesh,keepmesh=keepmesh)

        self.vecform = vecform
        """ Format for the vecs, 'XZ', 'XYZ', or 'Y'"""

        self.deformation = deformation
        """ Whether to include deformation potential effects
            in matrix elements"""

        self.piezo = piezo
        """ Whether to solve for the piezoelectric potential
            induced by the phonon and use it in matrix elements"""
    

    def u(self,iq,thetaq,l):
        """ The displacement profile

        Args:
            iq (int): index into the `q` array
            thetaq: in-plane angle of the phonon propagation (from X toward Y)
            l: which eigenvalue to use of those solved for

        Returns:
            a tuple of three PointFunctions (ux,uy,uz)

        """

        vec0=self.vecs[iq,l]
        if self.vecform=='XZ':
            ux= vec0[0,:]*np.cos(thetaq)
            uy= vec0[0,:]*np.sin(thetaq)
            uz= vec0[1,:]
        elif self.vecform=='XYZ':
            ux= vec0[0,:]*np.cos(thetaq) - vec0[1,:]*np.sin(thetaq)
            uy= vec0[0,:]*np.sin(thetaq) + vec0[1,:]*np.cos(thetaq)
            uz= vec0[2,:]
        elif self.vecform=='Y':
            ux=-vec0[0,:]*np.sin(thetaq)
            uy= vec0[0,:]*np.cos(thetaq)
            uz= vec0[0,:]*0

        return ux,uy,uz

    def strain(self,iq,thetaq,l):
        r""" The strain profile

        The strains returned are physicist strains, not engineering strains,
        eg :math:`e_{xy}=\frac{1}{2}\left(\partial_xu_y+\partial_yu_x\right)`

        Args:
            iq (int): index into the `q` array
            thetaq: in-plane angle of the phonon propagation (from X toward Y)
            l: which eigenvalue to use of those solved for

        Returns:
            a tuple of six MidFunctions (exx,exy,exz,eyy,eyz,ezz)

        """
        ux,uy,uz=self.u(iq,thetaq,l)
        q=self.q[iq]

        qx,qy=polar2cart(q,thetaq)
        exx=1j*qx*ux.tmf()
        exy=.5*1j*qx*uy.tmf()+.5*1j*qy*ux.tmf()
        exz=.5*1j*qx*uz.tmf()+.5*ux.differentiate()
        eyy=1j*qy*uy.tmf()
        eyz=.5*1j*qy*uz.tmf()+.5*uy.differentiate()
        ezz=uz.differentiate()

        return exx,exy,exz,eyy,eyz,ezz

    def I2(self,carrier,psii,psij,iq,thetaq,l):
        """ The matrix element squared between two wavefunctions

        See :func:`PhononModel.I2` for arguments and returns.

        Note: if both piezo and deformation potentials are included, they
        are combined coherently (ie *inside* the squaring).
        """

        I=0

        if self.deformation:
            exx,exy,exz,eyy,eyz,ezz=self.strain(iq,thetaq,l)
            D=self._keepmesh._matblocks[0].matsys.kp_strain_mat(self._keepmesh,
                exx=exx,exy=exy,exz=exz,eyy=eyy,eyz=eyz,ezz=ezz,carrier=carrier).tpf()
            psij_D_psii=complex(
                (np.sum(psij.conj().T*np.sum(np.rollaxis(D,-1,-2)*psii.T,axis=-1).T,axis=-1))\
                    .integrate(definite=True))
            I+=psij_D_psii
        if self.piezo:
            phi=self.rmesh['phi'][iq,l]
            psij_phi_psii=complex((np.sum(psij.conj()*phi*psii,axis=0)).integrate(definite=True))
            I+=psij_phi_psii

        return np.abs(I)**2


# TODO: Figure out how to move the glob_store _splines safely to superclass
@glob_store_attributes('_mesh','_keepmesh','_ec_load_matrix','rmesh','_splines')
class ElasticContinuum(AcousticPhonon):
    def __init__(self,mesh,rmesh,num_eigenvalues,keepmesh=None,
            vecform='XYZ',first_level=0,parallel=True,
            deformation=True,piezo=False):
        """ Note: this parallel is not the same as the one in solve()"""
        super().__init__(mesh,rmesh,vecform=vecform,keepmesh=keepmesh,
                deformation=deformation,piezo=piezo)
        m=mesh
        self._neig=num_eigenvalues
        self._first_level=first_level

        if vecform:
            self._n=len(vecform)

        assert len(mesh._matblocks)==1,\
            "ElasticContinuum only works on a mesh with a single material system for now"

        self._ec_load_matrix=assemble_load_matrix(w=m.density,dzp=m.dzp,n=self._n,
                dirichelet1=False,dirichelet2=False)

        if rmesh is not None:

            if 'en' in self.rmesh:
                self.rmesh['ref_en']=self.rmesh['en']
                del self.rmesh['en']

            if 'ec_stiffness_matrices' not in rmesh:
                log("Assembling EC matrices ...",level='info')
                if self.vecform=='XZ':
                    Cmats=m._matblocks[0].matsys.ec_CmatsXZ(m,self.q)
                elif self.vecform=='Y':
                    Cmats=m._matblocks[0].matsys.ec_CmatsY(m,self.q)
                else:
                    Cmats=m._matblocks[0].matsys.ec_Cmats(m,self.q)

                if parallel: pool=Pool.process_pool(new=True)
                else: pool=Pool.FakePool()
                self.rmesh['ec_stiffness_matrices']=pool.starmap(
                        assemble_stiffness_matrix,\
                            [(C0,Cl,Cr,C2,m._dzp,False,False)
                                for [C0,Cl,Cr,C2] in Cmats])
                log("Done assembly.",level='info')

        self.piezo=PiezoPotential(self,parallel=parallel) if piezo else None

    @property
    def _ec_stiffness_matrices(self): return self.rmesh['ec_stiffness_matrices']
    @property
    def phi(self): return self.rmesh['phi']


    def solve(self, just_energies=False, parallel=True, print_count=True):

        # Initialize other functions
        if 'en' not in self.rmesh:
            self.rmesh['en']   =np.empty((len(self.q),self._neig))
        if 'vecs' not in self.rmesh and not just_energies:
            self.rmesh['vecs'] =PointFunction(self._keepmesh,
                    empty=(len(self.q),self._neig,self._n),dtype='complex')
        if not just_energies and self.piezo and 'phi' not in self.rmesh:
            self.rmesh['phi'] =PointFunction(self._keepmesh,
                    empty=(len(self.q),self._neig),dtype='complex')

        counter=Counter(print_message="Count: {{:5d}}/{}".format(self.rmesh.N))
        def save_solve(iq,res):
            if just_energies:
                self.en[iq,:]= res
            elif not self.piezo:
                self.en[iq,:],self.vecs[iq,:,:,:]= res
            else:
                self.en[iq,:],self.vecs[iq,:,:,:],self.phi[iq,:,:]= res
            if print_count:
                counter.increment()
        pool=Pool.process_pool(new=True) if parallel else FakePool()
        asyncs=[pool.apply_async(self.solve_one_q,args=(None,iq,just_energies),
                callback=partial(save_solve,iq), error_callback=raiser)
                for iq in range(self.rmesh.N)]
        for asyn in asyncs: asyn.wait()

    def solve_one_q(self,q,iq=None,just_energies=False):
        m=self._mesh
        if iq is None:
            if self.vecform=='XZ':
                C0,Cl,Cr,C2=m._matblocks[0].matsys.ec_CmatsXZ(m,np.array([q]))[0]
            elif self.vecform=='Y':
                C0,Cl,Cr,C2=m._matblocks[0].matsys.ec_CmatsY( m,np.array([q]))[0]
            else:
                C0,Cl,Cr,C2=m._matblocks[0].matsys.ec_Cmats(  m,np.array([q]))[0]
            A=assemble_stiffness_matrix(C0,Cl,Cr,C2,m._dzp,dirichelet1=False,dirichelet2=False)
        else:
            A=self._ec_stiffness_matrices[iq]

        if self._first_level==0:
            mid_eig=0
            neig_ext=self._neig
        else:
            ref_en=self.rmesh['ref_en']
            mid_eig=(np.mean(ref_en[iq,self._first_level-1:self._first_level+1])/hbar)**2
            neig_ext=self._neig+6

        en_out=np.empty([neig_ext])
        vec_out=np.empty([neig_ext,self._n,m.Np],dtype=complex)\
            if not just_energies else False

        fem_eigsh(A,self._ec_load_matrix,en_out,vec_out,n=self._n,
             dirichelet1=False,dirichelet2=False,
             k=neig_ext,sigma=mid_eig-1e-10,which='LA',tol=0,ncv=max(neig_ext*2,neig_ext+2))
        en_out[:]=hbar*np.sqrt(en_out)

        if self._first_level!=0:
            ref_slice=slice(self._first_level,self._first_level+self._neig)
            for ioffset in [0,1,-1,2,-2,3,-3,None]:
                assert ioffset is not None, "Couldn't match reference energies"\
                    +" iq "+str(iq) + "\nen\n"+str(en_out)+"\nref\n"+\
                        str(ref_en[iq,ref_slice])
                off_slice=slice((3+ioffset),(neig_ext-3+ioffset))
                if np.allclose(ref_en[iq,ref_slice],en_out[off_slice],atol=1e-6): break
        else:
            off_slice=slice(None)

        en_out=en_out[off_slice]
        
        if just_energies:
            return en_out
        else:
            vec_out=(vec_out[off_slice].T*np.sqrt(hbar**2/(2*en_out))).T

            if not self.piezo:
                return en_out,\
                    PointFunction(m,vec_out,dtype='complex')\
                        .restrict(self._keepmesh)
            else:
                return en_out,\
                    PointFunction(m,vec_out,dtype='complex')\
                        .restrict(self._keepmesh),\
                    self.piezo.solve_one_q(q,iq,vec_out)\
                        .restrict(self._keepmesh)

@glob_store_attributes('_mesh','_keepmesh','rmesh')
class PiezoPotential():
    def __init__(self,pm,parallel=True):
        self.pm=pm
        self.rmesh=rmesh=pm.rmesh
        self.vecform=pm.vecform
        self._neig=pm._neig

        m=self._mesh=pm._mesh
        self._keepmesh=pm._keepmesh

        if rmesh and ('pz_stiffness_matrices' not in rmesh):
            log("Assembling PZ matrices ...",level='info')
            if parallel: pool=Pool.process_pool(new=True)
            else: pool=Pool.FakePool()
            self._eps_x=np.expand_dims(np.expand_dims(m.eps,0),0)
            self._eps_z=np.expand_dims(np.expand_dims(m.epsperp,0),0)
            self._O    =np.expand_dims(np.expand_dims(m.zeros_mid,0),0)
            self._e51=np.expand_dims(np.expand_dims(m.e51,0),0)
            self._e31=np.expand_dims(np.expand_dims(m.e31,0),0)
            self._e33=np.expand_dims(np.expand_dims(m.e33,0),0)

            self.rmesh['pz_stiffness_matrices']=pool.starmap(
                    assemble_stiffness_matrix,\
                        [(q**2*self._eps_x,None,None,self._eps_z,m._dzp,True,True)
                            for q in self.q])
            self.rmesh['pz_load_matrices_z']=pool.starmap(
                    assemble_stiffness_matrix,\
                        [(q**2*self._e51,None,None,self._e33,m._dzp,False,False)
                            for q in self.q])
            self.rmesh['pz_load_matrices_x']=pool.starmap(
                    assemble_stiffness_matrix,\
                        [(self._O,q*self._e51,q*self._e31,self._O,m._dzp,False,False)
                            for q in self.q])
            log("Done assembly.",level='info')
    
    @property
    def q(self): return self.rmesh.absk1

    def solve_one_q(self,q,iq,vec):
        m=self._mesh
    
        # Purely transverse modes have no piezo potential
        if self.vecform=='Y':
            return PointFunction(m,np.zeros((self._neig,m.Np),dtype='complex'))

        if iq is None:
            A_pz =assemble_stiffness_matrix(
                q**2*self._eps_x,None,None,self._eps_z,m._dzp,True,True)
            Mz_pz=assemble_stiffness_matrix(q**2*self._e51,None,None,self._e33,
                    m._dzp,False,False)
            Mx_pz=assemble_stiffness_matrix(self._O,q*self._e51,q*self._e31,self._O,
                    m._dzp,False,False)
        else:
            A_pz =self.rmesh['pz_stiffness_matrices'][iq]
            Mx_pz=self.rmesh['pz_load_matrices_x'][iq]
            Mz_pz=self.rmesh['pz_load_matrices_z'][iq]

        phi=PointFunction(m,empty=(self._neig,),dtype='complex')
        vslice=slice(1,-1) # dirichelet top, neumann bottom
        for e in range(self._neig):
            b_pz=(Mx_pz @ vec[e,0])[vslice] 
            if 'Z' in self.vecform:
                b_pz+=(Mz_pz @ vec[e,-1])[vslice]
            fem_solve(A_pz,None,b_pz,phi[e],1,True,True)
        return phi

class OpticalPhonon(PhononModel):
    def __init__(self, mesh, rmesh, keepmesh=None):
        super().__init__(mesh,rmesh,keepmesh=keepmesh)

    def I2(self,carrier,psii,psij,iq,thetaq,l):
        """ The matrix element squared between two wavefunctions

        See :func:`PhononModel.I2` for arguments and returns.

        Note:`thetaq` does not actually matter for optical phonons.
        """
        phi=self.rmesh['phi'][iq,l]
        psij_phi_psii=complex((np.sum(psij.conj()*phi*psii,axis=0)).integrate(definite=True))
        I=psij_phi_psii
        return np.abs(I)**2

class DielectricContinuum_SWH(OpticalPhonon):
    def __init__(self, mesh, rmesh, num_specific_eigenvalues, num_eigenvalues=None,first_level=0, keepmesh=None):
        """ Solves for the extraordinary polar optical phonons in a Single Wurtzite Heterojunction.

        See the Dielectric Continuum :ref:`BWH` model for the relevant mathematics.

        Args:
            mesh: The mesh to solve on, should contain one :class:`~pynitride.material.Wurtzite` material block, which
                has two layers of uniform molefraction.
            rmesh: The :class:`~pynitride.reciprocal_mesh.RMesh_1D` which specifies the :math:`q` points
            num_specific_eigenvalues: should be a dictionary indicating how many eigenvalues are desired for each mode
                type, ie mapping the names `'TOl','TOIF','TOu','LOl','LOIF','LOu'` to integers.  `'l','IF','u'` refer to
                the lower-region confined, interface, and upper-region confined modes respectively
            num_eigenvalues: if specified, only this many contiguous eigenvalues will be used out of the modes specified
                by `num_specific_eigenvalues`
            first_level: can be used in combination with `num_eigenvalues` to select which set of `num_eigenvalues`
                eigenvalues will be used.
            keepmesh: the mesh on which to actually store the solved `phi`

        """

        super().__init__(mesh, rmesh, vecform=None, keepmesh=keepmesh)

        # Requirements for a Heterojunction
        assert len(mesh._matblocks) == 1, \
            "ElasticContinuum only works on a mesh with a single material system for now"
        assert isinstance(mesh._matblocks[0].matsys, AlGaN)
        assert len(mesh._layers) == 2

        # Get the meshes for the upper and lower layers
        self._umesh = umesh = mesh._layers[0].mesh
        self._lmesh = lmesh = mesh._layers[1].mesh

        # Get the LO frequencies for the upper and lower layers
        wLO_perp_u = umesh.wLO_perp[0]
        wLO_para_u = umesh.wLO_para[0]
        wLO_perp_l = lmesh.wLO_perp[0]
        wLO_para_l = lmesh.wLO_para[0]

        # Get the TO frequencies for the upper and lower layers
        wTO_perp_u = umesh.wTO_perp[0]
        wTO_para_u = umesh.wTO_para[0]
        wTO_perp_l = lmesh.wTO_perp[0]
        wTO_para_l = lmesh.wTO_para[0]

        # Get the high-frequency dielectric constants
        epsinf_u = umesh.eps_inf[0]
        epsinf_l = lmesh.eps_inf[0]

        # Get the thicknesses
        t1 = mesh._layers[0].thickness
        t2 = mesh._layers[1].thickness

        # Compile all the above into an array for quick reference in helper functions
        self._params = [wLO_perp_u, wLO_para_u, wLO_perp_l, wLO_para_l,
                        wTO_perp_u, wTO_para_u, wTO_perp_l, wTO_para_l,
                        epsinf_u, epsinf_l, t1, t2]

        # Which layer is the lower-frequency one
        self._slowlayer = mesh._layers[int(wLO_perp_l < wLO_perp_u)]
        self._fastlayer = mesh._layers[int(wLO_perp_l > wLO_perp_u)]

        # Make sure the frequencies are ordered as we expect
        assert np.all(np.diff([
                self._slowlayer.mesh.wTO_para[0],
                self._slowlayer.mesh.wTO_perp[0],
                self._fastlayer.mesh.wTO_para[0],
                self._fastlayer.mesh.wTO_perp[0],
                self._slowlayer.mesh.wLO_para[0],
                self._slowlayer.mesh.wLO_perp[0],
                self._fastlayer.mesh.wLO_para[0],
                self._fastlayer.mesh.wLO_perp[0]])>0),\
            "Characteristic POP frequencies are not ordered as expected."

        # Whether the u or l modes appear first depends on which material is u/l
        regs_order=['u','IF','l'] \
            if self._slowlayer==self._mesh._layers[0] else \
            ['l','IF','u']
        self.mode_order=[p+r for p,r in product(['TO','LO'],regs_order)]

        # Incorporate the information of num_specific_eigenvalues, num_eigenvalues, and first_level to
        # figure out exactly how many and which of each mode type to include
        self._neig=OrderedDict()
        self._firstlevels=OrderedDict()
        neig_sofar=0
        neig_included_sofar=0
        num_eigenvalues_max=num_eigenvalues if num_eigenvalues is not None else np.infty
        for m in self.mode_order:

            # Number of eigenvalues we're allowed to pull from this type of mode
            navail=num_specific_eigenvalues[m]

            # not including any levels that would fall below first_level
            spec_first_level=self._firstlevels[m]=max(first_level-neig_sofar,0)
            navail_highenough=max(navail-spec_first_level,0)

            # not including any levels that would fall above first_level+num_eigenvalues
            navail_highenough_lowenough=min(num_eigenvalues_max-neig_included_sofar,navail_highenough)

            # include these values
            self._neig[m]=navail_highenough_lowenough
            neig_sofar+=num_specific_eigenvalues[m]
            neig_included_sofar+=navail_highenough_lowenough
        assert num_eigenvalues is None or num_eigenvalues==neig_included_sofar

        # Select the correct set of energies and modes from rmesh if a fuller set is already present from a supersolve
        if 'en' in self.rmesh:
            self.rmesh['ref_en']=self.rmesh['en']
            self.rmesh['en']=self.rmesh['ref_en'][:,first_level:first_level+num_eigenvalues]
        if 'phi' in self.rmesh:
            self.rmesh['phi']=self.rmesh['phi'][:,first_level:first_level+num_eigenvalues]

    @property
    def phi(self):
        """ The 3-D array of potentials, shape `(len(self.q),sum(self._neig.values()),self._mesh.Np)`."""
        return self.rmesh['phi']

    def get_mode_by_name(self,name,num,iq=None):
        """ Convenience function to pull particular modes from the `phi` array by name.

        Args:
            name (str): name of the mode type, eg `'TOu'` for the TO mode confined to the upper region
            num: which of the modes solved for to return (indexed from 0 being the first solved-for mode of this type)
            iq: if specified, will return only for the given :math:`q` index (may be integer or slice)
        Returns:
            a tuple of the energy(ies), potential(s)
        """
        assert num<self._neig[name], "Requested "+str(num)+"-th "+name+" mode, which was not solved for"

        lmin=([0]+list(np.cumsum([self._neig[n] for n in self._neig.keys()])))[list(self._neig.keys()).index(name)]
        l=lmin+num
        if iq is None: iq=slice(None)
        return self.en[iq,l],self.phi[iq,l,:]

    def solve(self, just_energies=False):
        """ Actually solve for the modes."""

        # Can only do a mode solve after an energy solve
        if not just_energies:
            if 'en' not in self.rmesh:
                self.solve(just_energies=True)


        # Make energy array if needed
        if 'en' not in self.rmesh:
            self.rmesh['en']   =np.empty((len(self.q),sum(self._neig.values())))

        # Make phi array if needed
        if not just_energies and 'phi' not in self.rmesh:
            self.rmesh['phi'] =PointFunction(self._keepmesh,
                 empty=(len(self.q),sum(self._neig.values())),dtype='double')

        lmin=0

        for (modetype, neig), fl in zip(self._neig.items(),self._firstlevels.values()):
            lmax=lmin+neig
            if just_energies:
                w=getattr(self,'_reg_'+modetype[2:])(self.q,pol=modetype[0],num=neig+fl)
                self.rmesh['en'][:,lmin:lmax]=hbar*w[:,fl:]
            else:
                en=self.rmesh['en'][:,lmin:lmax]
                for iq in range(len(self.q)):
                    for iw in range(neig):
                       self.rmesh['phi'][iq,lmin+iw,:]=\
                           self._get_mode(self.q[iq],en[iq,iw]/hbar,reg=modetype[2:])\
                               .restrict(self._keepmesh)
            lmin=lmax

    def _common(self, w):
        """ Evaluates some variables needed frequently throughout the math for many functions."""
        wLO_perp_u, wLO_para_u, wLO_perp_l, wLO_para_l, \
        wTO_perp_u, wTO_para_u, wTO_perp_l, wTO_para_l, \
        epsinf_u, epsinf_l, t1, t2 = self._params

        eps_perp_u = epsinf_u * (wLO_perp_u ** 2 - w ** 2) / (wTO_perp_u ** 2 - w ** 2)
        eps_para_u = epsinf_u * (wLO_para_u ** 2 - w ** 2) / (wTO_para_u ** 2 - w ** 2)
        eps_perp_l = epsinf_l * (wLO_perp_l ** 2 - w ** 2) / (wTO_perp_l ** 2 - w ** 2)
        eps_para_l = epsinf_l * (wLO_para_l ** 2 - w ** 2) / (wTO_para_l ** 2 - w ** 2)

        xi_u = np.sqrt(np.abs(eps_perp_u * eps_para_u))
        xi_l = np.sqrt(np.abs(eps_perp_l * eps_para_l))

        alpha_u = np.sqrt(np.abs(eps_perp_u / eps_para_u))
        alpha_l = np.sqrt(np.abs(eps_perp_l / eps_para_l))

        return eps_perp_u, eps_para_u, eps_perp_l, eps_para_l, xi_u, xi_l, alpha_u, alpha_l

    def _reg_u(self, q, pol='T', num=30):
        """ Solves for energies of upper-region confined modes.

        Args:
            q: the q to solve at
            pol: the polarization 'T' or 'L'
            num: the number of energies to return
        Returns:
            an array of energies, shape `(len(q),num)`
        """
        wLO_perp_u, wLO_para_u, wLO_perp_l, wLO_para_l, \
        wTO_perp_u, wTO_para_u, wTO_perp_l, wTO_para_l, \
        epsinf_u, epsinf_l, t1, t2 = self._params

        wmin, wmax = (wTO_para_u, wTO_perp_u) if pol == 'T' else (wLO_para_u, wLO_perp_u)
        wmin += 1e-7 * meV / hbar;
        wmax -= 1e-7 * meV / hbar;
        wtest = np.linspace(wmin, wmax, 500000)

        eps_perp_u, eps_para_u, eps_perp_l, eps_para_l, xi_u, xi_l, alpha_u, alpha_l = self._common(wtest)
        s = np.sign(eps_para_u[0] * eps_para_l[0])
        qtest = 1 / (alpha_u * t1) * (np.arctan(-s * xi_u / xi_l) + np.expand_dims(np.arange(num + 1), 1) * pi)
        if np.max(qtest[0, :]) < np.max(q):
            qtest = qtest[1:, :]
        else:
            qtest = qtest[:-1, :]

        w = []
        for qtesti in qtest:
            w += [interp1d(qtesti, wtest)(q)]
        w = np.array(w).T

        return w

    def w_IF(self, pol='T'):
        """ Finds the inteface resonant frequency.

        Args:
            pol: the polarization 'T' or 'L'
        Returns:
            a tuple of the interface resonant frequency and +1/-1 indicating the mode is
            found above/below this frequency respectively
        """

        wTO_perp_G = self._slowlayer.mesh.wTO_perp[0]
        wTO_para_A = self._fastlayer.mesh.wTO_para[0]
        wLO_perp_G = self._slowlayer.mesh.wLO_perp[0]
        wLO_para_A = self._fastlayer.mesh.wLO_para[0]

        wmin, wmax = (wTO_perp_G, wTO_para_A) if pol == 'T' else (wLO_perp_G, wLO_para_A)
        wmin += 1e-5 * meV / hbar;
        wmax -= 1e-5 * meV / hbar;

        def xi_l_minus_xi_u(w):
            xi_u, xi_l = self._common(w)[4:6]
            return xi_l - xi_u

        wres = brentq(xi_l_minus_xi_u, wmin, wmax)

        return wres, np.sign(xi_l_minus_xi_u((wres + wmax) / 2))

    def _reg_IF(self, q, pol='T',num=1):
        """ Solves for energies of interface modes.

        Args:
            q: the q to solve at
            pol: the polarization 'T' or 'L'
            num: the number of energies to return, at most 1
        Returns:
            an array of energies, shape `(len(q),num)`
        """
        assert num in [0,1], "There's only one "+pol\
                             +"OIF mode, don't ask for more!"
        wLO_perp_u, wLO_para_u, wLO_perp_l, wLO_para_l, \
        wTO_perp_u, wTO_para_u, wTO_perp_l, wTO_para_l, \
        epsinf_u, epsinf_l, t1, t2 = self._params

        wTO_perp_G = self._slowlayer.mesh.wTO_perp[0]
        wTO_para_A = self._fastlayer.mesh.wTO_para[0]
        wLO_perp_G = self._slowlayer.mesh.wLO_perp[0]
        wLO_para_A = self._fastlayer.mesh.wLO_para[0]

        wres, side = self.w_IF(pol)
        if side < 0:
            wmin, wmax = (wTO_perp_G, wres) if pol == 'T' else (wLO_perp_G, wres)
        else:
            wmin, wmax = (wres, wTO_para_A) if pol == 'T' else (wres, wLO_para_A)
        wmin += 1e-5 * meV / hbar;
        wmax -= 1e-5 * meV / hbar;
        wtest = np.linspace(wmin, wmax, 10000)

        eps_perp_u, eps_para_u, eps_perp_l, eps_para_l, xi_u, xi_l, alpha_u, alpha_l = self._common(wtest)
        qtest = 1 / (2 * alpha_u * t1) * np.log((xi_l + xi_u) / (xi_l - xi_u))
        w = np.expand_dims(interp1d(qtest, wtest, fill_value=(np.NaN, wres), bounds_error=False)(q), -1)
        return w

    def _reg_l(self, q, pol='T', num=100):
        """ Solves for energies of lower-region confined modes.

        Args:
            q: the q to solve at
            pol: the polarization 'T' or 'L'
            num: the number of energies to return
        Returns:
            an array of energies, shape `(len(q),num)`
        """
        wLO_perp_u, wLO_para_u, wLO_perp_l, wLO_para_l, \
        wTO_perp_u, wTO_para_u, wTO_perp_l, wTO_para_l, \
        epsinf_u, epsinf_l, t1, t2 = self._params

        wmin, wmax = (wTO_para_l, wTO_perp_l) if pol == 'T' else (wLO_para_l, wLO_perp_l)
        wmin += 1e-6 * meV / hbar;
        wmax -= 1e-6 * meV / hbar;
        wtest = np.linspace(wmin, wmax, 100000)

        w = []
        alpha_l = self._common(wtest)[7]
        for n in range(num):
            k2 = pi * (n + 1) / t2
            qtest = k2 / alpha_l
            w += [interp1d(qtest, wtest)(q)]
        return np.array(w).T

    def _get_mode(self, q, w, reg):
        r""" Produces the analytic mode given the already solved position in :math:`(q,\omega)`.

        Args:
            q: the in-plane wavevector
            w: the angular frequency
            reg: 'u','IF','l' indicating where the mode is (upper/interface/lower)

        Returns:
            the potential as a PointFunction on the `keepmesh`
        """
        wLO_perp_u, wLO_para_u, wLO_perp_l, wLO_para_l, \
        wTO_perp_u, wTO_para_u, wTO_perp_l, wTO_para_l, \
        epsinf_u, epsinf_l, t1, t2 = self._params

        eps_perp_u, eps_para_u, eps_perp_l, eps_para_l, xi_u, xi_l, alpha_u, alpha_l = self._common(w)
        k_u = q * alpha_u;
        k_l = q * alpha_l

        ew2_u = epsinf_u * ((wLO_para_u ** 2 - wTO_para_u ** 2) + (wLO_perp_u ** 2 - wTO_perp_u ** 2)) / 2
        ew2_l = epsinf_l * ((wLO_para_l ** 2 - wTO_para_l ** 2) + (wLO_perp_l ** 2 - wTO_perp_l ** 2)) / 2

        beta2_para_u = ew2_u * (k_u / (wTO_para_u ** 2 - w ** 2)) ** 2
        beta2_perp_u = ew2_u * (q / (wTO_perp_u ** 2 - w ** 2)) ** 2
        beta2_para_l = ew2_l * (k_l / (wTO_para_l ** 2 - w ** 2)) ** 2
        beta2_perp_l = ew2_l * (q / (wTO_perp_l ** 2 - w ** 2)) ** 2

        if reg == 'u':
            gamma2_para_u = 1 / 2 * (t1 + 1 / (2 * k_u) * np.sin(2 * k_u * t1))
            gamma2_perp_u = 1 / 2 * (t1 - 1 / (2 * k_u) * np.sin(2 * k_u * t1))
            gamma2_para_l = 1 / (2 * k_l) * np.exp(-2 * k_l * t1)
            gamma2_perp_l = 1 / (2 * k_l) * np.exp(-2 * k_l * t1)
            BoA = np.sin(k_u * t1) * np.exp(k_l * t1)
        if reg == 'IF':
            gamma2_para_u = 1 / 2 * (1 / (2 * k_u) * np.sinh(2 * k_u * t1) + t1)
            gamma2_perp_u = 1 / 2 * (1 / (2 * k_u) * np.sinh(2 * k_u * t1) - t1)
            gamma2_para_l = 1 / (2 * k_l) * np.exp(-2 * k_l * t1)
            gamma2_perp_l = 1 / (2 * k_l) * np.exp(-2 * k_l * t1)
            BoA = np.sinh(k_u * t1) * np.exp(k_l * t1)
        if reg == 'l':
            gamma2_para_u = 1 / 2 * (1 / (2 * k_u) * np.sinh(2 * k_u * t1) + t1)
            gamma2_perp_u = 1 / 2 * (1 / (2 * k_u) * np.sinh(2 * k_u * t1) - t1)
            gamma2_para_l = t2 / 2
            gamma2_perp_l = t2 / 2
            s = np.sign(eps_para_u * eps_para_l)
            theta = np.arctan(s * xi_l / xi_u * np.tanh(k_u * t1)) - k_l * t1
            BoA = np.sinh(k_u * t1) / np.sin(k_l * t1 + theta)

        A = np.sqrt(
            hbar / (2 * w) / (beta2_para_u * gamma2_para_u + beta2_perp_u * gamma2_perp_u +
                              (BoA) ** 2 * (beta2_para_l * gamma2_para_l + beta2_perp_l * gamma2_perp_l)))
        B = BoA * A
        phi_ = PointFunction(self._keepmesh, empty=())
        phi = phi_.restrict(self._keepmesh._matblocks[0].mesh)
        if reg == 'u':
            phi.restrict(self._umesh)[:] = A * np.sin(k_u * self._umesh.zp)
            phi.restrict(self._lmesh)[:] = B * np.exp(-k_l * self._lmesh.zp)
        if reg == 'IF':
            phi.restrict(self._umesh)[:] = A * np.sinh(k_u * self._umesh.zp)
            phi.restrict(self._lmesh)[:] = B * np.exp(-k_l * self._lmesh.zp)
        if reg == 'l':
            phi.restrict(self._umesh)[:] = A * np.sinh(k_u * self._umesh.zp)
            phi.restrict(self._lmesh)[:] = B * np.sin(k_l * self._lmesh.zp + theta)
        return phi_


