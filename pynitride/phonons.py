import numpy as np
from pynitride.cython.assemblers.assemble3x3 import assemble3x3
from pynitride.visual import log, sublog
from scipy.sparse.linalg import eigsh
from pynitride.mesh import PointFunction
from pynitride.paramdb import hbar
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
    def __init__(self,mesh,num_eigenvalues,qmax,num_qpoints,qshift=0,bctop=.0001,bcbottom=.0001):
        super().__init__(mesh)
        self._qmax=qmax
        self._num_qpoints=num_qpoints
        self._neig=num_eigenvalues
        self._qshift=qshift
        self._bctop=bctop
        self._bcbottom=bcbottom

        assert len(mesh._matblocks)==1, "ElasticContinuum only works on a mesh with a single material system for now"
        mesh.request_function("C11")

    def initialize(self):
        m=self._mesh
        qmax=self._qmax
        num_qpoints=self._num_qpoints
        num_eigenvalues=self._neig

        #q=self._q=np.flipud(np.linspace(qmax,0,num_qpoints,endpoint=False))
        q=self._q=np.linspace(0,qmax,num_qpoints,endpoint=True)
        q+=self._qshift
        #q[0]+=qmax/(num_qpoints-1)*.01
        Cmats=m._matblocks[0].matsys.ec_Cmats(m,q)
        log("Assembling EC matrices ...",level='info')
        self._H=[assemble3x3(C0,Cl,Cr,C2,m._dzm,m._dzp,bctop=m.ztrans*self._bctop,bcbottom=m.ztrans*self._bcbottom) for [C0,Cl,Cr,C2] in Cmats]
        log("Done assembly.",level='info')
        self._en   =np.empty((len(self._H),self._neig))
        self._vecs =PointFunction(m,empty=(len(self._H),self._neig,3),dtype='complex')

    def solve(self,debugmode=False):
        m=self._mesh
        log("Phonon Solve",level="debug")
        q=self._q
        if debugmode:
            self._eigenvectorsH=[]
        for i,(qi,H) in enumerate(zip(q,self._H)):
            eigenvals,eigenvectors=eigsh(H,k=self._neig,sigma=0,which='LM',tol=0,ncv=self._neig*2)
            indarr=np.argsort(eigenvals)
            if debugmode:
                self._eigenvectorsH+=[eigenvectors[:,indarr]]
            self._en[i,:]=hbar*np.sqrt(eigenvals[indarr])
            self._vecs[i,:,:,:]=np.swapaxes(np.reshape(
                eigenvectors[:,indarr],
                (len(m._zp),3,self._neig)),0,2) \
                                /np.sqrt(m._dzm)
