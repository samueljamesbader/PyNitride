import numpy as np
from pynitride.machine import Pool, glob_store_attributes
from pynitride.visual import log, sublog
from scipy.sparse.linalg import eigsh
from pynitride.mesh import PointFunction
from pynitride.paramdb import hbar
from pynitride.fem import assemble_stiffness_matrix, assemble_load_matrix, fem_eigsh
from functools import partial

class PhononModel():
    """ Superclass for all phonon models.

    A phonon model implements a :func:`solve()` function
    """

    def __init__(self, mesh):
        """ Prep the mesh for any phonon model.
        """
        self._mesh=mesh

    def solve(self):
        raise NotImplementedError

@glob_store_attributes('_mesh','_load_matrix','_stiffness_matrices','rmesh')
class ElasticContinuum(PhononModel):
    def __init__(self,mesh,rmesh,num_eigenvalues,justXZ=False):
        super().__init__(mesh)
        m=mesh
        self.rmesh=rmesh
        self._neig=num_eigenvalues
        self._justXZ=justXZ
        self._n=3-justXZ

        assert len(mesh._matblocks)==1, "ElasticContinuum only works on a mesh with a single material system for now"

        self._load_matrix=assemble_load_matrix(w=m.density,dzp=m.dzp,n=self._n,dirichelet1=False,dirichelet2=False)

        if rmesh is not None:


            log("Assembling EC matrices ...",level='info')
            if self._justXZ:
                Cmats=m._matblocks[0].matsys.ec_CmatsXZ(m,self.q)
            else:
                Cmats=m._matblocks[0].matsys.ec_Cmats(m,self.q)
            self._stiffness_matrices=[assemble_stiffness_matrix(C0,Cl,Cr,C2,m._dzp,dirichelet1=False,dirichelet2=False)
                        for [C0,Cl,Cr,C2] in Cmats]
            log("Done assembly.",level='info')

            # Initialize other functions
            if 'en' not in self.rmesh:
                self.rmesh['en']   =np.empty((len(self.q),self._neig))
                self.rmesh['vecs'] =PointFunction(m,empty=(len(self.q),self._neig,self._n),dtype='complex')

    @property
    def q(self): return self.rmesh.absk1
    @property
    def en(self): return self.rmesh['en']
    @property
    def vecs(self): return self.rmesh['vecs']

    def solve(self):
        def save_solve(iq,res):
            self.en[iq,:],self.vecs[iq,:,:,:]= res
        pool=Pool.process_pool(new=True)
        asyncs=[pool.apply_async(self.solve_one_q,args=(None,iq),
                                 callback=partial(save_solve,iq))
                for iq in range(self.rmesh.N)]
        for async in asyncs: async.wait()

    def solve_one_q(self,q,iq=None):
        m=self._mesh
        if iq is None:
            C0,Cl,Cr,C2=m._matblocks[0].matsys.ec_Cmats(m,np.array([q]))[0]
            A=assemble_stiffness_matrix(C0,Cl,Cr,C2,m._dzp,dirichelet1=False,dirichelet2=False)
        else:
            A=self._stiffness_matrices[iq]

        en_out=np.empty([self._neig])
        vec_out=np.empty([self._neig,self._n,m.Np],dtype=complex)
        fem_eigsh(A,self._load_matrix,en_out,vec_out,n=self._n,
                                         dirichelet1=False,dirichelet2=False,
                                         k=self._neig,sigma=0,which='LM',tol=0,ncv=self._neig*2)
        en_out[:]=hbar*np.sqrt(en_out)
        vec_out[:]=(vec_out.T*np.sqrt(hbar**2/(2*en_out))).T
        return en_out,vec_out

