from pynitride.machine import Pool, glob_store_attributes, FakePool, Counter, raiser
from pynitride.visual import log, sublog
from pynitride.mesh import PointFunction
from pynitride.paramdb import pmdb,hbar, meV
from pynitride.fem import assemble_stiffness_matrix, assemble_load_matrix, fem_eigsh, fem_solve
from pynitride.maths import polar2cart
from pynitride.material import AlGaN
from scipy.sparse.linalg import eigsh
from functools import partial
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from itertools import product
from operator import itemgetter
from collections import OrderedDict
import numpy as np
pi=np.pi

class PhononModel():

    def __init__(self, solvmesh, rmesh, num_eigs, keepmesh=None, first_level=0):
        """ Superclass for all phonon models.
        
        Args:
            solvmesh: the direct mesh on which to perform the phonon solve
            rmesh: the 1-D reciprocal mesh on which to perform the phonon solve
            num_eigs: the number of eigenvalues to solve for
            keepmesh: the direct mesh on which phonon modes will actually be
                desired, eg for a scattering problem, just the region where the
                carriers are.  This should be a submesh of `solvmesh`.
            first_level: the index of the first eigenvalue which should be
                solved for.
        """

        self._solvmesh=solvmesh
        self._keepmesh=solvmesh if keepmesh is None else keepmesh

        self.rmesh=rmesh
        """ The reciprocal mesh on which to solve"""

        self.num_eigs=num_eigs
        """ The number of eigenvalues to solve for"""

        self.first_level=first_level
        """ The index of the first eigenvalue to solve for"""

        self._interp_ready=False

    def solve(self, mode_iqs=None):
        pass

    @property
    def q(self):
        """ The q values of the `self.rmesh`."""
        return self.rmesh.absk1

    @property
    def _en(self):
        return self.rmesh['en']
    @property
    def _vecs(self):
        return self.rmesh['vecs']
    @property
    def _phi(self):
        return self.rmesh['phi']

    def _check_l_index(self,l):
        # Checks that l is between first_level and first_level+num_eigs
        assert l is None or\
                (l>=self.first_level and l<=self.first_level+self.num_eigs),\
            "l index {} out of bounds {}-{}".format(
                    l,self.first_level,self.first_level+self.num_eigs)

    def en(self,iq=slice(None),l=None):
        """ Returns the `l`-th energy at q-index `iq`.

        Note: `l` is an absolute index,
        ie `l` should not be lower than first_level

        """
        self._check_l_index(l)
        return self._en[iq,slice(None) if l is None else l-self.first_level]

    def vecs(self,iq=slice(None),l=None):
        """ Returns the `l`-th mode vector at q-index `iq`.

        Note: `l` is an absolute index,
        ie `l` should not be lower than first_level

        """
        self._check_l_index(l)
        return self._vecs[iq,slice(None) if l is None else l-self.first_level]

    def phi(self,iq=slice(None),l=slice(None)):
        """ Returns the `l`-th potential at q-index `iq`.

        Note: `l` is an absolute index,
        ie `l` should not be lower than first_level

        """
        self._check_l_index(l)
        return self._phi[iq,slice(None) if l is None else l-self.first_level]

    _save_with_energies=['en']
    _save_with_vecs=['en','vecs','phi']
    def save(self,filename,just_energies=False):
        """ Saves the phonon solve to a file.

        Args:
            filename (str): the file to save to
            just_energies (bool): if True, save only the energies,
                otherwise save energies, mode vecs and/or potentials.
        """
        if just_energies:
            keys= self._save_with_energies
        else:
            keys=[k for k in self._save_with_vecs if k in self.rmesh]
        self.rmesh.save(filename,keys=keys)

    def read(self,name,just_energies=False):
        """ Reads the phonon solve from a file.

        If reading mode vectors and/or potentials, checks the dimensions.
        If dimensions are incorrect, any new keys in rmesh
        will be cleared away and then an error will be raised.

        Args:
            filename (str): the file to save to
            just_energies (bool): if True, read only the energies,
                otherwise read energies, mode vecs and/or potentials.
        """
        try:
            if just_energies:
                self.rmesh.read(name,keys=self._save_with_energies)
            else:
                self.rmesh.read(name)
                assert 'vecs' in self.rmesh or 'phi' in self.rmesh

            if 'en' in self.rmesh:
                assert self._en.shape==(self.rmesh.N,self.num_eigs),\
                    "Loaded PhononModel does not match current"
            if 'vecs' in self.rmesh:
                assert self._vecs.shape==\
                    (self.rmesh.N,self.num_eigs,self._n,self._keepmesh.Np),\
                    "Loaded PhononModel does not match current"
                self.rmesh['vecs']=PointFunction(self._keepmesh,self._vecs,
                        dtype=self._vecs.dtype)
            if 'phi' in self.rmesh:
                assert self._phi.shape==\
                    (self.rmesh.N,self.num_eigs,self._keepmesh.Np),\
                    "Loaded PhononModel does not match current"
                self.rmesh['phi']=PointFunction(self._keepmesh,self._phi,
                        dtype=self._phi.dtype)
        except:
            for key in self._save_with_vecs:
                if key in self.rmesh: del self.rmesh[key]
            raise


    def _get_interpolation(self):
        # Preps self._splines which will hold interpolated energy functions
        if not self._interp_ready:
            self._splines=[self.rmesh.interpolator(self.rmesh['en'][:,eig])
                        for eig in range(self.num_eigs)]
            self._interp_ready=True

    def interp_energy(self,absk,l,bounds_check=True):
        """ Returns the interpolated energy at a point in reciprocal space

        Args:
            absk (float): the point in reciprocal space
            l (int): the mode index (absolute)
            bounds_check (bool): whether to complain if absk is out of
                interpolation bounds
        """
        self._check_l_index(l)
        self._get_interpolation()
        return self._splines[l-self.first_level]\
                (absk,bounds_check=bounds_check)

    def interp_radial_group_velocity(self,absk,l,bounds_check=True):
        """ Returns the interpolated radial group velocity at a point in reciprocal space

        Args:
            absk (float): the point in reciprocal space
            l (int): the mode index (absolute)
            bounds_check (bool): whether to complain if absk is out of
                interpolation bounds
        """
        self._check_l_index(l)
        self._get_interpolation()
        return 1/hbar*self._splines[l-self.first_level]\
                (absk,dabsk=1,bounds_check=bounds_check)

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
    
    def __init__(self,solvmesh,rmesh,num_eigs,keepmesh=None,first_level=0,
            vecform='XYZ',deformation=True,piezo=False):
        r""" Superclass for acoustic phonons.

        Args:
            solvmesh,rmesh,keepmesh, first_level: see :class:`~PhononModel`
            vecform (str): Format for the vecs, 'XZ', 'XYZ', or 'Y'
            deformation (bool): whether to include deformation potential
                effects in matrix elements
            piezo (bool): whether to solve for the piezoelectric potential
                induced by the phonon and use it in matrix elements

        """
        super().__init__(solvmesh,rmesh,num_eigs=num_eigs,
                keepmesh=keepmesh,first_level=first_level)

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

        vec0=self.vecs(iq,l)
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
            phi=self.phi(iq,l)
            psij_phi_psii=complex((np.sum(psij.conj()*phi*psii,axis=0)).integrate(definite=True))
            I+=psij_phi_psii

        return np.abs(I)**2


# TODO: Figure out how to move the glob_store _splines safely to superclass
@glob_store_attributes('_solvmesh','_keepmesh','_ec_load_matrix','rmesh','_splines')
class ElasticContinuum(AcousticPhonon):
    def __init__(self,solvmesh,rmesh,num_eigs,keepmesh=None,
            vecform='XYZ',first_level=0,parallel=True,
            deformation=True,piezo=False,dirichelet_bottom=False):
        """ Note: this parallel is not the same as the one in solve()"""
        super().__init__(solvmesh,rmesh,num_eigs=num_eigs,
                keepmesh=keepmesh,first_level=first_level,
                vecform=vecform,deformation=deformation,piezo=piezo)
        m=solvmesh
        self.num_eigs=num_eigs

        self._n=len(vecform)
        self._dbot=dirichelet_bottom

        assert len(m._matblocks)==1,\
            "ElasticContinuum only works on a mesh with a single material system for now"

        self._ec_load_matrix=assemble_load_matrix(w=m.density,dzp=m.dzp,n=self._n,
                dirichelet1=False,dirichelet2=self._dbot)

        if rmesh is not None:

            if 'en' in self.rmesh:
                self.rmesh['ref_en']=self.rmesh['en']
                self.rmesh['en']=self.rmesh['ref_en']\
                        [:,self.first_level:self.first_level+self.num_eigs]
            if 'vecs' in self.rmesh:
                self.rmesh['vecs']=self.rmesh['vecs']\
                        [:,self.first_level:self.first_level+self.num_eigs,:,:]
            if 'phi' in self.rmesh:
                self.rmesh['phi']=self.rmesh['phi']\
                        [:,self.first_level:self.first_level+self.num_eigs,:]

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
                            [(C0,Cl,Cr,C2,m._dzp,False,self._dbot)
                                for [C0,Cl,Cr,C2] in Cmats])
                log("Done assembly.",level='info')

        self.piezo=PiezoPotential(self,parallel=parallel) if piezo else None

    @property
    def _ec_stiffness_matrices(self): return self.rmesh['ec_stiffness_matrices']

    def solve(self, just_energies=False, parallel=True, print_count=True,mode_iqs=None):

        # Initialize other functions
        if 'en' not in self.rmesh:
            self.rmesh['en']   =np.empty((len(self.q),self.num_eigs))
        if 'vecs' not in self.rmesh and not just_energies:
            self.rmesh['vecs'] =PointFunction(self._keepmesh,
                    empty=(len(self.q),self.num_eigs,self._n),dtype='complex')
        if not just_energies and self.piezo and 'phi' not in self.rmesh:
            self.rmesh['phi'] =PointFunction(self._keepmesh,
                    empty=(len(self.q),self.num_eigs),dtype='complex')

        counter=Counter(print_message="Count: {{:5d}}/{}".format(self.rmesh.N))
        def save_solve(iq,res):
            if just_energies:
                self._en[iq,:]= res
            elif not self.piezo:
                self._en[iq,:],self._vecs[iq,:,:,:]= res
            else:
                self._en[iq,:],self._vecs[iq,:,:,:],self._phi[iq,:,:]= res
            if print_count:
                counter.increment()
        pool=Pool.process_pool(new=True) if parallel else FakePool()
        asyncs=[pool.apply_async(self.solve_one_q,args=(None,iq,just_energies),
                callback=partial(save_solve,iq), error_callback=raiser)
                for iq in range(self.rmesh.N)]
        for asyn in asyncs: asyn.wait()

    def solve_one_q(self,q,iq=None,just_energies=False):
        m=self._solvmesh
        if iq is None:
            if self.vecform=='XZ':
                C0,Cl,Cr,C2=m._matblocks[0].matsys.ec_CmatsXZ(m,np.array([q]))[0]
            elif self.vecform=='Y':
                C0,Cl,Cr,C2=m._matblocks[0].matsys.ec_CmatsY( m,np.array([q]))[0]
            else:
                C0,Cl,Cr,C2=m._matblocks[0].matsys.ec_Cmats(  m,np.array([q]))[0]
            A=assemble_stiffness_matrix(C0,Cl,Cr,C2,m._dzp,
                    dirichelet1=False,dirichelet2=self._dbot)
        else:
            A=self._ec_stiffness_matrices[iq]

        if self.first_level==0:
            mid_eig=0
            neig_ext=self.num_eigs
        else:
            ref_en=self.rmesh['ref_en']
            mid_eig=(np.mean(ref_en[iq,self.first_level-1:self.first_level+1])/hbar)**2
            neig_ext=self.num_eigs+6

        en_out=np.empty([neig_ext])
        vec_out=np.empty([neig_ext,self._n,m.Np],dtype=complex)\
            if not just_energies else False

        fem_eigsh(A,self._ec_load_matrix,en_out,vec_out,n=self._n,
             dirichelet1=False,dirichelet2=self._dbot,
             k=neig_ext,sigma=mid_eig-1e-10,which='LA',tol=0,ncv=max(neig_ext*2,neig_ext+2))
        en_out[:]=hbar*np.sqrt(en_out)

        if self.first_level!=0:
            ref_slice=slice(self.first_level,self.first_level+self.num_eigs)
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

@glob_store_attributes('_solvmesh','_keepmesh','rmesh')
class PiezoPotential():
    def __init__(self,pm,parallel=True):
        self.pm=pm
        self.rmesh=rmesh=pm.rmesh
        self.vecform=pm.vecform
        self.num_eigs=pm.num_eigs

        self._solvmesh=m=pm._solvmesh
        self._keepmesh=  pm._keepmesh

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
        m=self._solvmesh
    
        # Purely transverse modes have no piezo potential
        if self.vecform=='Y':
            return PointFunction(m,np.zeros((self.num_eigs,m.Np),dtype='complex'))

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

        phi=PointFunction(m,empty=(self.num_eigs,),dtype='complex')
        vslice=slice(1,-1) # dirichelet top, neumann bottom
        for e in range(self.num_eigs):
            b_pz=(Mx_pz @ vec[e,0])[vslice] 
            if 'Z' in self.vecform:
                b_pz+=(Mz_pz @ vec[e,-1])[vslice]
            fem_solve(A_pz,None,b_pz,phi[e],1,True,True)
        return phi

class OpticalPhonon(PhononModel):
    def __init__(self, solvmesh, rmesh, num_eigs, keepmesh=None, first_level=0):
        super().__init__(solvmesh,rmesh,num_eigs=num_eigs, keepmesh=keepmesh, first_level=first_level)

    def I2(self,carrier,psii,psij,iq,thetaq,l):
        """ The matrix element squared between two wavefunctions

        See :func:`PhononModel.I2` for arguments and returns.

        Note:`thetaq` does not actually matter for optical phonons.
        """
        phi=self.phi(iq,l)
        psij_phi_psii=complex((np.sum(psij.conj()*phi*psii,axis=0)).integrate(definite=True))
        I=psij_phi_psii
        return np.abs(I)**2

# TODO: Figure out how to move the glob_store _splines safely to superclass
@glob_store_attributes('_solvmesh','_keepmesh','rmesh','_splines')
class ElasticContinuum_BulkWurtzite(AcousticPhonon):

    def __init__(self,solvmesh,rmesh,num_eigs,
            thickness,matname,
            keepmesh=None,first_level=0,vecform='XYZ',polXZ='all'):
        super().__init__(solvmesh=solvmesh,rmesh=rmesh,num_eigs=num_eigs,
                first_level=first_level,vecform=vecform,keepmesh=keepmesh)
        m=self._keepmesh
        self._n=len(self.vecform)
        self._pol=polXZ
        
        self._thickness=thickness
        self._c44=pmdb['material={}.stiffness.C44'.format(matname)]
        self._c11=pmdb['material={}.stiffness.C11'.format(matname)]
        self._c12=pmdb['material={}.stiffness.C12'.format(matname)]
        self._c13=pmdb['material={}.stiffness.C13'.format(matname)]
        self._c33=pmdb['material={}.stiffness.C33'.format(matname)]
        self._rho=pmdb['material={}.density'.format(matname)]
        
        if 'en' in self.rmesh:
            self.rmesh['ref_en']=self.rmesh['en']
            self.rmesh['en']=self.rmesh['ref_en']\
                [:,self.first_level:self.first_level+self.num_eigs]
        if 'beta' in self.rmesh:
            self.rmesh['beta']=self.rmesh['beta']\
                [:,self.first_level:self.first_level+self.num_eigs]
        if 'modetype' in self.rmesh:
            self.rmesh['modetype']=self.rmesh['modetype']\
                [:,self.first_level:self.first_level+self.num_eigs]
        if 'vecs' in self.rmesh:
            self.rmesh['vecs']=self.rmesh['vecs']\
                [:,self.first_level:self.first_level+self.num_eigs,:,:]
        if 'phi' in self.rmesh:
            self.rmesh['phi']=self.rmesh['phi']\
                [:,self.first_level:self.first_level+self.num_eigs,:]

    _save_with_energies=['en','beta','modetype']
    _save_with_vecs=['en','beta','modetype','vecs','phi']
    @property
    def _beta(self): return self.rmesh['beta']
    @property
    def _modetype(self): return self.rmesh['modetype']

    def solve(self, just_energies=False, print_count=False,mode_iqs=None):

        # Can only do a mode solve after an energy solve
        if 'en' not in self.rmesh:
            self._solve_energies()

        # All the energy work is already done by _solve_energies
        if just_energies: return

        # Make vecs array if needed
        if 'vecs' not in self.rmesh:
            self.rmesh['vecs'] =PointFunction(self._keepmesh,
                 empty=(len(self.q),self.num_eigs,self._n),dtype='complex')

        # Make phi array if needed
        if self.piezo and 'phi' not in self.rmesh:
            self.rmesh['phi'] =PointFunction(self._keepmesh,
                 empty=(len(self.q),self.num_eigs),dtype='complex')
        
        # Parameters
        c44,c11,c12,c13,c33,rho=itemgetter(
            '_c44','_c11','_c12','_c13','_c33','_rho')(self.__dict__)

        # Ratio of Z to X in an XZ mode
        def d_XZ(q,beta,w):
            # Assumes beta,q !=0
            with np.errstate(divide='ignore',invalid='ignore'):
                return (-beta**2*c44-c11*q**2+w**2*rho)/(beta*q*(c13+c44))

        # At each q
        for iq in range(self.rmesh.N):
            q=self.q[iq]

            # Make a components array for each mode
            comps=np.zeros((self.num_eigs,self._n))

            # Y modes just have Y component = 1
            if self.vecform=='Y':
                comps[:,:]=1
            elif self.vecform=='XYZ':
                masky =(self._modetype[iq]=='Y')
                comps[masky,1]=1

            # XZ modes split XZ by the above d_XZ
            if 'X' in self.vecform:
                maskxz=(self._modetype[iq]!='Y')
                comps[maskxz,0] =1
                comps[maskxz,-1]=d_XZ(q,self._beta[iq,maskxz],self._en[iq,maskxz]/hbar)
                for iw in np.arange(self.num_eigs)[(self._beta[iq,:]==0) & maskxz]:
                    comps[iw,[0,-1]]={'LA':[1,0],'TA':[0,1]}[self._modetype[iq,iw]]

            # Renormalize components to scattering-appropriate normalization 
            w=self._en[iq,:]/hbar
            comps/= np.atleast_2d(np.sqrt(self._thickness*rho*np.sum(np.abs(comps)**2\
                                /(hbar/(2*np.atleast_2d(w).T)),axis=1))).T

            # Get vectors as product of e^(i beta z) and component weighting
            comps=np.atleast_3d(comps)
            self._vecs[iq,:,:]=\
                np.exp(1j*np.swapaxes(np.atleast_3d(self._beta[iq,:]),0,1)\
                    *self._keepmesh.zp)*comps

    def _solve_energies(self):
        print("in _solve_energies")

        # Make energy array if needed
        if 'en' not in self.rmesh:
            self.rmesh['en']      =np.empty((len(self.q),self.num_eigs))
            self.rmesh['beta']    =np.empty((len(self.q),self.num_eigs))
            self.rmesh['modetype']=np.empty((len(self.q),self.num_eigs),
                                        dtype='object')
        
        # Fast enough to make parallelization silly
        for iq in range(self.rmesh.N):
            self._solve_one_energy(iq)

    def _solve_one_energy(self,iq):
        assert self.first_level==0
        q=self.q[iq]

        # Parameters
        c44,c11,c12,c13,c33,rho=itemgetter(
            '_c44','_c11','_c12','_c13','_c33','_rho')(self.__dict__)
         
        # Dispersion relation for XZ modes
        def _w_pm(pm,q,beta):
            D4,D2,D0=[
                rho**2,
                -beta**2*(c33+c44)*rho-(c11+c44)*rho*q**2,
                beta**4*c33*c44 + beta**2*c11*c33*q**2 - beta**2*c13**2*q**2\
                    - 2*beta**2*c13*c44*q**2 + c11*c44*q**4]
            return np.sqrt((-D2+pm*np.sqrt(D2**2-4*D4*D0))/(2*D4))
        w_LA=partial(_w_pm,+1)
        w_TA=partial(_w_pm,-1)

        # Dispersion relation for Y modes
        vY=np.sqrt((c11-c12)/(2*rho))
        w_Y = lambda q,beta: vY*np.sqrt(q**2+beta**2)

        # Get first num_eigs energies for each mode type
        n=np.concatenate([np.arange(- int(self.num_eigs/2),1),
                          np.arange(1,int(self.num_eigs/2)+1)])      
        beta=2*pi*n/self._thickness
        doY =('Y' in self.vecform)
        doLA=('X' in self.vecform and self._pol is not 'TA')
        doTA=('X' in self.vecform and self._pol is not 'LA')
        w=np.concatenate([w_Y( q,beta)  if doY else [],
                          w_TA(q,beta)  if doTA else [],
                          w_LA(q,beta)  if doLA else []])
        
        # Identify which are collectively the lowest num_eigs 
        modetype=(['Y'] *self.num_eigs\
                        if doY else [])\
                + (['TA']*self.num_eigs\
                        if doTA else [])\
                + (['LA']*self.num_eigs\
                        if doLA else [])
        iw=np.argsort(w)[:self.num_eigs]
        self._en[iq,:]=hbar*w[iw]
        self._modetype[iq,:]=np.array(modetype)[iw]
        self._beta[iq,:]=np.tile(beta,self._n)[iw]


@glob_store_attributes('_solvmesh','_keepmesh','rmesh','_umesh','_lmesh','_slowlayer','_fastlayer','_splines')
class DielectricContinuum_SWH(OpticalPhonon):
    def __init__(self, solvmesh, rmesh, num_spec_eigs, num_eigs=None,first_level=0, keepmesh=None):
        """ Solves for the extraordinary PO phonons in a Single Wurtzite Heterojunction.

        See the Dielectric Continuum :ref:`BWH` model for the relevant mathematics.

        Args:
            solvmesh: The mesh to solve on, should contain one :class:`~pynitride.material.Wurtzite` material block, which
                has two layers of uniform molefraction.
            rmesh: The :class:`~pynitride.reciprocal_mesh.RMesh_1D` which specifies the :math:`q` points
            num_spec_eigs: should be a dictionary indicating how many eigenvalues are desired for each mode
                type, ie mapping the names `'TOl','TOIF','TOu','LOl','LOIF','LOu'` to integers.  `'l','IF','u'` refer to
                the lower-region confined, interface, and upper-region confined modes respectively
            num_eigs: if specified, only this many contiguous eigenvalues will be used out of the modes specified
                by `num_spec_eigs`
            first_level: can be used in combination with `num_eigs` to select which set of `num_eigs`
                eigenvalues will be used.
            keepmesh: the mesh on which to actually store the solved `phi`

        """

        super().__init__(solvmesh, rmesh, keepmesh=keepmesh, num_eigs=num_eigs, first_level=first_level)
        mesh=self._solvmesh

        # Requirements for a Heterojunction
        assert len(mesh._matblocks) == 1, \
            "DielectricContinuum_SWH only works on a mesh with a single material block"
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
            if self._slowlayer==self._solvmesh._layers[0] else \
            ['l','IF','u']
        self.mode_order=[p+r for p,r in product(['TO','LO'],regs_order)]

        # Incorporate the information of num_spec_eigs, num_eigs, and first_level to
        # figure out exactly how many and which of each mode type to include
        self._neig=OrderedDict()
        self._firstlevels=OrderedDict()
        neig_sofar=0
        neig_included_sofar=0
        num_eigs_max=num_eigs if num_eigs is not None else np.infty
        for m in self.mode_order:

            # Number of eigenvalues we're allowed to pull from this type of mode
            navail=num_spec_eigs[m]

            # not including any levels that would fall below first_level
            spec_first_level=self._firstlevels[m]=max(first_level-neig_sofar,0)
            navail_highenough=max(navail-spec_first_level,0)

            # not including any levels that would fall above first_level+num_eigs
            navail_highenough_lowenough=min(num_eigs_max-neig_included_sofar,navail_highenough)

            # include these values
            self._neig[m]=navail_highenough_lowenough
            neig_sofar+=num_spec_eigs[m]
            neig_included_sofar+=navail_highenough_lowenough
        assert num_eigs is None or num_eigs==neig_included_sofar
        self.num_eigs=neig_included_sofar

        # Select the correct set of energies and modes from rmesh if a fuller set is already present from a supersolve
        if 'en' in self.rmesh:
            self.rmesh['ref_en']=self.rmesh['en']
            self.rmesh['en']=self.rmesh['ref_en'][:,first_level:first_level+num_eigs]
        if 'phi' in self.rmesh:
            self.rmesh['phi']=self.rmesh['phi'][:,first_level:first_level+num_eigs]

    def get_mode_by_name(self,name,num,iq=None):
        """ Convenience function to pull particular modes from the `phi` array by name.

        Args:
            name (str): name of the mode type,
                eg `'TOu'` for the TO mode confined to the upper region
            num: which of the modes solved for to return
                (indexed from 0 being the first solved-for mode of this type)
            iq: if specified, will return only for the given :math:`q` index
                (may be integer or slice)
        Returns:
            a tuple of the energy(ies), potential(s)
        """
        assert num>=self._firstlevels[name] and num<self._neig[name]+self._firstlevels[name],\
            "Requested "+str(num)+"-th "+name+" mode, which was not solved for"

        lmin=([0]+list(np.cumsum([self._neig[n]\
                for n in self._neig.keys()])))\
            [list(self._neig.keys()).index(name)]
        l=lmin+num
        if iq is None: iq=slice(None)
        return self._en[iq,l],self._phi[iq,l,:]

    def solve(self, just_energies=False, print_count=False, mode_iqs=None):
        """ Actually solve for the modes."""

        assert not print_count
        iqs=mode_iqs if mode_iqs is not None else range(len(self.q))

        # Can only do a mode solve after an energy solve
        if not just_energies:
            if 'en' not in self.rmesh:
                self.solve(just_energies=True)


        # Make energy array if needed
        if 'en' not in self.rmesh:
            self.rmesh['en']   =np.empty((len(self.q),self.num_eigs))

        # Make phi array if needed
        if not just_energies and 'phi' not in self.rmesh:
            self.rmesh['phi'] =PointFunction(self._keepmesh,
                 empty=(len(self.q),self.num_eigs),dtype='double')

        lmin=0

        for (modetype, neig), fl in zip(self._neig.items(),self._firstlevels.values()):
            if neig==0: continue
            lmax=lmin+neig
            if just_energies:
                w=getattr(self,'_reg_'+modetype[2:])(self.q,pol=modetype[0],num=neig+fl)
                self.rmesh['en'][:,lmin:lmax]=hbar*w[:,fl:]
            else:
                en=self.rmesh['en'][:,lmin:lmax]
                for iq in iqs:
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

        #TODO: make this actually limit to keepmesh
        phi_ = PointFunction(self._solvmesh, empty=())
        phi = phi_.restrict(self._solvmesh._matblocks[0].mesh)
        if reg == 'u':
            phi.restrict(self._umesh)[:] = A * np.sin(k_u * self._umesh.zp)
            phi.restrict(self._lmesh)[:] = B * np.exp(-k_l * self._lmesh.zp)
        if reg == 'IF':
            phi.restrict(self._umesh)[:] = A * np.sinh(k_u * self._umesh.zp)
            phi.restrict(self._lmesh)[:] = B * np.exp(-k_l * self._lmesh.zp)
        if reg == 'l':
            phi.restrict(self._umesh)[:] = A * np.sinh(k_u * self._umesh.zp)
            phi.restrict(self._lmesh)[:] = B * np.sin(k_l * self._lmesh.zp + theta)
        return phi_.restrict(self._keepmesh)





# TODO: Figure out how to move the glob_store _splines safely to superclass
@glob_store_attributes('_solvmesh','_keepmesh','rmesh','_splines')
class DielectricContinuum_BulkWurtzite(OpticalPhonon):

    def __init__(self,solvmesh,rmesh,num_eigs,
            thickness,matname,
            keepmesh=None,first_level=0,pol='L'):
        super().__init__(solvmesh=solvmesh,rmesh=rmesh,num_eigs=num_eigs,
                first_level=first_level,keepmesh=keepmesh)
        m=self._keepmesh
        self._pol=pol
        
        self._thickness=thickness
        self._eps_inf =pmdb['material={}.dielectric.eps_inf'.format(matname)]
        self._wLO_para=pmdb['material={}.raman.wLO_para'    .format(matname)]
        self._wLO_perp=pmdb['material={}.raman.wLO_perp'    .format(matname)]
        self._wTO_para=pmdb['material={}.raman.wTO_para'    .format(matname)]
        self._wTO_perp=pmdb['material={}.raman.wTO_perp'    .format(matname)]
        
        if 'en' in self.rmesh:
            self.rmesh['ref_en']=self.rmesh['en']
            self.rmesh['en']=self.rmesh['ref_en']\
                [:,self.first_level:self.first_level+self.num_eigs]
        if 'beta' in self.rmesh:
            self.rmesh['beta']=self.rmesh['beta']\
                [:,self.first_level:self.first_level+self.num_eigs]
        #if 'modetype' in self.rmesh:
        #    self.rmesh['modetype']=self.rmesh['modetype']\
        #        [:,self.first_level:self.first_level+self.num_eigs]
        if 'phi' in self.rmesh:
            self.rmesh['phi']=self.rmesh['phi']\
                [:,self.first_level:self.first_level+self.num_eigs,:]

    _save_with_energies=['en','beta','modetype']
    _save_with_vecs=['en','beta','modetype','vecs','phi']
    @property
    def _beta(self): return self.rmesh['beta']
    @property
    def _modetype(self): return self.rmesh['modetype']

    def solve(self, just_energies=False, print_count=False,mode_iqs=None):

        # Can only do a mode solve after an energy solve
        if 'en' not in self.rmesh:
            self._solve_energies()

        # All the energy work is already done by _solve_energies
        if just_energies: return


        # Make phi array if needed
        if 'phi' not in self.rmesh:
            self.rmesh['phi'] =PointFunction(self._keepmesh,
                 empty=(len(self.q),self.num_eigs),dtype='complex')

        # Parameters
        wLO_para,wLO_perp,wTO_para,wTO_perp,eps_inf=itemgetter(
            '_wLO_para','_wLO_perp','_wTO_para','_wTO_perp','_eps_inf')(self.__dict__)
        phi=np.exp(1j*np.expand_dims(self._beta,-1)*self._keepmesh.zp)
        w=self._en/hbar
        ew2 = eps_inf * ((wLO_para ** 2 - wTO_para ** 2) + (wLO_perp ** 2 - wTO_perp ** 2)) / 2
        Nint=(self._thickness*ew2*((self._beta.T/(wTO_para**2-w.T**2))**2+(self.q/(wTO_perp**2-w.T**2))**2)).T
        Nreq=hbar/(2*w)
        self._phi[:,:,:]=(phi.T/np.sqrt(Nint/Nreq).T).T


    def _solve_energies(self):
        assert self.first_level==0

        # Make energy array if needed
        if 'en' not in self.rmesh:
            self.rmesh['en']      =np.empty((len(self.q),self.num_eigs))
            self.rmesh['beta']    =np.empty((len(self.q),self.num_eigs))

        # Parameters
        wLO_para,wLO_perp,wTO_para,wTO_perp,eps_inf=itemgetter(
            '_wLO_para','_wLO_perp','_wTO_para','_wTO_perp','_eps_inf')(self.__dict__)

        # Bounded by the relevant para/perp frequency
        wbounds=(wLO_para,wLO_perp) if self._pol=='L' else (wTO_para,wTO_perp)
        wmin,wmax=np.sort(wbounds)+[1e-7*meV,-1e-7*meV]
        wtest=np.linspace(wmin,wmax,10000)

        def _common(w):
            eps_perp = eps_inf * (wLO_perp ** 2 - w ** 2) / (wTO_perp ** 2 - w ** 2)
            eps_para = eps_inf * (wLO_para ** 2 - w ** 2) / (wTO_para ** 2 - w ** 2)
            alpha = np.sqrt(np.abs(eps_perp / eps_para))
            return eps_perp, eps_para, alpha
        eps_perp,eps_para,alpha=_common(wtest)

        # Get all quantized z momenta, excludes zero mode
        n=np.ravel(np.tile(
            np.arange(1,int(self.num_eigs/2)+2),(2,1)).T\
                *[-1,1])[:self.num_eigs]
        beta=2*pi*n/self._thickness

        # Interpolate to get energy versus q
        for i,betai in enumerate(beta):
            qtest=alpha*np.abs(betai)
            #print("qtest range",np.min(qtest),np.max(qtest))
            w=interp1d(qtest,wtest)(self.q)
            self._en[:,i]=hbar*w
        self._beta[:,:]=beta

