import numpy as np
from pynitride.machine import Pool, glob_store_attributes, FakePool, Counter, raiser
from pynitride.visual import log, sublog
from scipy.sparse.linalg import eigsh
from pynitride.mesh import PointFunction
from pynitride.paramdb import hbar
from pynitride.fem import assemble_stiffness_matrix, assemble_load_matrix, fem_eigsh, fem_solve
from pynitride.maths import polar2cart
from functools import partial

class PhononModel():
    """ Superclass for all phonon models.

    A phonon model implements a :func:`solve()` function
    """

    def __init__(self, mesh, rmesh, vecform, keepmesh=None):
        """ Prep the mesh for any phonon model.
        """
        self._mesh=mesh
        self._keepmesh=mesh if keepmesh is None else keepmesh
        self.rmesh=rmesh
        self._interp_ready=False

        self.vecform = vecform
        """ Format for the vecs, 'XZ', 'XYZ', or 'Y'"""

        self._n=len(vecform)

    def solve(self):
        raise NotImplementedError

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


    def u(self,i,thetaq,l):

        vec0=self.vecs[i,l]
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

    def strain(self,i,thetaq,l):
        ux,uy,uz=self.u(i,thetaq,l)
        q=self.q[i]

        qx,qy=polar2cart(q,thetaq)
        exx=1j*qx*ux.tmf()
        exy=.5*1j*qx*uy.tmf()+.5*1j*qy*ux.tmf()
        exz=.5*1j*qx*uz.tmf()+.5*ux.differentiate()
        eyy=1j*qy*uy.tmf()
        eyz=.5*1j*qy*uz.tmf()+.5*uy.differentiate()
        ezz=uz.differentiate()

        return exx,exy,exz,eyy,eyz,ezz


# TODO: Figure out how to move the glob_store _splines safely to superclass
@glob_store_attributes('_mesh','_keepmesh','_ec_load_matrix','rmesh','_splines')
class ElasticContinuum(PhononModel):
    def __init__(self,mesh,rmesh,num_eigenvalues,keepmesh=None,
            vecform='XYZ',first_level=0,parallel=True,piezo_potential=False):
        """ Note: this parallel is not the same as the one in solve()"""
        super().__init__(mesh,rmesh,vecform,keepmesh=keepmesh)
        m=mesh
        self._neig=num_eigenvalues
        self._first_level=first_level

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

        self._piezo=PiezoPotential(self,parallel=parallel)\
            if piezo_potential else None

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
        if not just_energies and self._piezo and 'phi' not in self.rmesh:
            self.rmesh['phi'] =PointFunction(self._keepmesh,
                    empty=(len(self.q),self._neig),dtype='complex')

        counter=Counter(print_message="Count: {{:5d}}/{}".format(self.rmesh.N))
        def save_solve(iq,res):
            if just_energies:
                self.en[iq,:]= res
            elif not self._piezo:
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
            C0,Cl,Cr,C2=m._matblocks[0].matsys.ec_Cmats(m,np.array([q]))[0]
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

            if not self._piezo:
                return en_out,\
                    PointFunction(m,vec_out,dtype='complex')\
                        .restrict(self._keepmesh)
            else:
                return en_out,\
                    PointFunction(m,vec_out,dtype='complex')\
                        .restrict(self._keepmesh),\
                    self._piezo.solve_one_q(q,iq,vec_out)\
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
            self.rmesh['pz_load_matrices_x']=pool.starmap(
                    assemble_stiffness_matrix,\
                        [(q**2*self._e51,None,None,self._e33,m._dzp,False,False)
                            for q in self.q])
            self.rmesh['pz_load_matrices_z']=pool.starmap(
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
                q**2*self._eps_x,None,None,self._epsz,m._dzp,True,True)
            Mx_pz=assemble_stiffness_matrix(q**2*self._e51,None,None,self._e33,
                    m._dzp,False,False)
            Mz_pz=assemble_stiffness_matrix(self._O,q*self._e51,q*self._e31,self._O,
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

