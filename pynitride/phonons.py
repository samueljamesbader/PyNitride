import numpy as np
from pynitride.visual import log, sublog
from scipy.sparse.linalg import eigsh
from pynitride.mesh import PointFunction
from pynitride.paramdb import hbar
from pynitride.fem import assemble_stiffness_matrix, assemble_load_matrix, fem_eigsh

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

class ElasticContinuum(PhononModel):
    def __init__(self,mesh,num_eigenvalues,qmax,num_qpoints,qshift=0, justXZ=False):
        super().__init__(mesh)
        m=mesh
        self._qmax=qmax
        self._num_qpoints=num_qpoints
        self._neig=num_eigenvalues
        self._qshift=qshift
        self._justXZ=justXZ
        self._n=3-justXZ

        assert len(mesh._matblocks)==1, "ElasticContinuum only works on a mesh with a single material system for now"

        self._load_matrix=assemble_load_matrix(w=m.density,dzp=m.dzp,n=self._n,dirichelet1=False,dirichelet2=False)

        if num_qpoints!=0:
            q=self._q=np.linspace(0,qmax,num_qpoints,endpoint=True)
            q+=self._qshift

            log("Assembling EC matrices ...",level='info')
            if self._justXZ:
                Cmats=m._matblocks[0].matsys.ec_CmatsXZ(m,q)
            else:
                Cmats=m._matblocks[0].matsys.ec_Cmats(m,q)
            self._stiffness_matrices=[assemble_stiffness_matrix(C0,Cl,Cr,C2,m._dzp,dirichelet1=False,dirichelet2=False)
                        for [C0,Cl,Cr,C2] in Cmats]
            log("Done assembly.",level='info')

            self._en   =np.empty((len(self._q),self._neig))
            self._vecs =PointFunction(m,empty=(len(self._q),self._neig,self._n),dtype='complex')

    def solve(self):
        m=self._mesh
        q=self._q
        for i,(qi,A) in enumerate(zip(q,self._stiffness_matrices)):
            self.solve_one_q(None,A,en_out=self._en[i,:],vec_out=self._vecs[i,:,:,:])

    def solve_one_q(self,q,A=None, en_out=None, vec_out=None):
        m=self._mesh
        if A is None:
            C0,Cl,Cr,C2=m._matblocks[0].matsys.ec_Cmats(m,np.array([q]))[0]
            A=assemble_stiffness_matrix(C0,Cl,Cr,C2,m._dzp,dirichelet1=False,dirichelet2=False)

        if en_out is None:
            en_out=np.empty([self._neig])
        if vec_out is None:
            vec_out=np.empty([self._neig,self._n,m.Np],dtype=complex)

        fem_eigsh(A,self._load_matrix,en_out,vec_out,n=self._n,
                                         dirichelet1=False,dirichelet2=False,
                                         k=self._neig,sigma=0,which='LM',tol=0,ncv=self._neig*2)
        en_out[:]=hbar*np.sqrt(en_out)
        vec_out[:]=(vec_out.T*np.sqrt(hbar**2/(2*en_out))).T
        return en_out,vec_out

