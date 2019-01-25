import numpy as np
from pynitride.machine import Pool, glob_store_attributes, FakePool, Counter, raiser
from pynitride.visual import log, sublog
from scipy.sparse.linalg import eigsh
from pynitride.mesh import PointFunction
from pynitride.paramdb import hbar
from pynitride.fem import assemble_stiffness_matrix, assemble_load_matrix, fem_eigsh
from pynitride.maths import polar2cart
from functools import partial

class PhononModel():
    """ Superclass for all phonon models.

    A phonon model implements a :func:`solve()` function
    """

    def __init__(self, mesh, rmesh):
        """ Prep the mesh for any phonon model.
        """
        self._mesh=mesh
        self.rmesh=rmesh
        self._interp_ready=False

        self.vecform = None
        """ Format for the vecs, 'XZ', 'XYZ', or 'Y'"""

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
    def read(self,name):
        self.rmesh.read(name)
        if 'en' in self.rmesh:
            assert self.en.shape==(self.rmesh.N,self._neig),\
                "Loaded PhononModel does not match current"
        if 'vecs' in self.rmesh:
            assert self.vecs.shape==(self.rmesh.N,self._neig,self._n,self._mesh.Np),\
                "Loaded PhononModel does not match current"
            self.rmesh['vecs']=PointFunction(self._mesh,self.vecs,dtype=self.vecs.dtype)


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
        else:
            raise NotImplementedError("Y Modes")

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
@glob_store_attributes('_mesh','_load_matrix','_stiffness_matrices','rmesh','_splines')
class ElasticContinuum(PhononModel):
    def __init__(self,mesh,rmesh,num_eigenvalues,justXZ=False,first_level=0,parallel=True):
        """ Note: this parallel is not the same as the one in solve()"""
        super().__init__(mesh,rmesh)
        m=mesh
        self._neig=num_eigenvalues
        self._justXZ=justXZ
        self._n=3-(justXZ is not False)
        self.vecform='XZ' if justXZ else 'XYZ'
        self._first_level=first_level

        assert len(mesh._matblocks)==1,\
            "ElasticContinuum only works on a mesh with a single material system for now"

        self._load_matrix=assemble_load_matrix(w=m.density,dzp=m.dzp,n=self._n,
                dirichelet1=False,dirichelet2=False)

        if rmesh is not None:

            if 'en' in self.rmesh:
                self.rmesh['ref_en']=self.rmesh['en']
                del self.rmesh['en']

            log("Assembling EC matrices ...",level='info')
            if self._justXZ:
                Cmats=m._matblocks[0].matsys.ec_CmatsXZ(m,self.q)
            else:
                Cmats=m._matblocks[0].matsys.ec_Cmats(m,self.q)
            self._stiffness_matrices=Pool.process_pool(new=True).starmap(
                    assemble_stiffness_matrix,\
                        [(C0,Cl,Cr,C2,m._dzp,False,False)
                            for [C0,Cl,Cr,C2] in Cmats])
            log("Done assembly.",level='info')


    def solve(self, just_energies=False, parallel=True):

        # Initialize other functions
        if 'en' not in self.rmesh:
            self.rmesh['en']   =np.empty((len(self.q),self._neig))
        if 'vecs' not in self.rmesh and not just_energies:
            self.rmesh['vecs'] =PointFunction(self._mesh,
                    empty=(len(self.q),self._neig,self._n),dtype='complex')

        counter=Counter(print_message="Count: {{:5d}}/{}".format(self.rmesh.N))
        def save_solve(iq,res):
            if just_energies:
                self.en[iq,:]= res
            else:
                self.en[iq,:],self.vecs[iq,:,:,:]= res
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
            A=self._stiffness_matrices[iq]

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

        fem_eigsh(A,self._load_matrix,en_out,vec_out,n=self._n,
             dirichelet1=False,dirichelet2=False,
             k=neig_ext,sigma=mid_eig-1e-10,which='LA',tol=0,ncv=neig_ext*2)
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

        if just_energies:
            return en_out[off_slice]
        else:
            vec_out[:]=(vec_out.T*np.sqrt(hbar**2/(2*en_out))).T
            return en_out[off_slice],vec_out[off_slice]

