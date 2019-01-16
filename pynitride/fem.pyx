cimport cython
cimport numpy as cnp
import numpy as np
from libc.math cimport pow, exp, sqrt
from scipy.sparse import lil_matrix
from functools import wraps
cnp.import_array()
from cpython cimport bool
from scipy.sparse.linalg import eigsh, spsolve

ctypedef fused CMat:
    cnp.ndarray[cnp.complex128_t,ndim=3]
    cnp.ndarray[cnp.float64_t,ndim=3]



#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.cdivision(True)
cpdef assemble_stiffness_matrix(
        CMat C0, CMat Cl,
        CMat Cr, CMat C2,
        cnp.ndarray[cnp.float64_t   ,ndim=1] dzp,
        bool dirichelet1=True, bool dirichelet2=True):
    """ Assemble a stiffness matrix from the coefficients of the differential equation.
    
    Cythonized assembly method, call as `assemble_stiffness_matrix(C0, Cl, Cr, C2, dzp, dirichelet1, dirichelet2)`.
    
    For the mathematics/definitions of the :math:`C` terms, see :ref:`FEM`.
    All arguments should be defined on the mid mesh except :math:`U`,
    and all the :math:`C` matrices should be three-dimensional, even if they are just scalars.
    
    If `Cl` and `Cr` are supplied as None, they will be ignored and the matrix will be of dtype double.
    If `Cl` and `Cr` are supplied, the matrix will be of dtype complex.
    
    Args:
        C0,Cl,Cr,C2: material-dependent differential equation coefficients
        dzp: the spacings between mesh nodes
        dirichelet1,dirichelet2 (bool): whether to treat boundary 1, 2 as Dirichelet (True, default) or Neumann.
    
    Returns:
        A stiffness matrix in sparse CSC form
    """
    cdef:
        int n, i,j,r,nz,z, zmat
        list rinds, rdata
        bool complex
    n =C0.shape[0]
    nz=dzp.shape[0]+1
    complex = (Cl is not None)
    lil=lil_matrix((n*(nz-dirichelet1-dirichelet2),n*(nz-dirichelet1-dirichelet2)),
                   dtype='complex' if complex else 'float')
    for z in range(dirichelet1,nz-dirichelet2):
        for i in range(n):
            zmat=z-dirichelet1
            r=zmat*n+i
            rinds=lil.rows[r]
            rdata=lil.data[r]

            #put in left
            if zmat>0:
                rinds+=range(n*zmat-n,n*zmat)
                rdata+=[0]*n
                for j in range(n):
                    rdata[j]= \
                        +C0[i,j,z-1]*dzp[z-1]/6\
                        -C2[i,j,z-1]/dzp[z-1]
                    if complex:rdata[j]+= \
                        +.5j*(Cl[i,j,z-1]+Cr[i,j,z-1])
            #put in diag
            rinds+=range(n*zmat,n*zmat+n)
            rdata+=[0]*n
            if z>0:
                for j in range(n):
                    rdata[(zmat>0)*n+j]+= \
                        +C0[i,j,z-1]*dzp[z-1]/3 \
                        +C2[i,j,z-1]/dzp[z-1]
                    if complex: rdata[(zmat>0)*n+j]+= \
                        +.5j*(-Cl[i,j,z-1]+Cr[i,j,z-1])
            if z<nz-1:
                for j in range(n):
                    rdata[(zmat>0)*n+j]+= \
                        +C0[i,j,z]*dzp[z]/3 \
                        +C2[i,j,z]/dzp[z]
                    if complex: rdata[(zmat>0)*n+j]+= \
                        +.5j*(+Cl[i,j,z]-Cr[i,j,z])\
            #put in right
            if z<nz-1-dirichelet2:
                rinds+=range(n*zmat+n,n*zmat+2*n)
                rdata+=[0]*n
                for j in range(n):
                    rdata[(1+(zmat>0))*n+j]= \
                        +C0[i,j,z]*dzp[z]/6\
                        -C2[i,j,z]/dzp[z]
                    if complex: rdata[(1+(zmat>0))*n+j]+= \
                        -.5j*(Cl[i,j,z]+Cr[i,j,z])
    return lil.asformat('csc')




#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.cdivision(True)
cpdef assemble_load_matrix(
        cnp.ndarray[cnp.float64_t   ,ndim=1] w,
        cnp.ndarray[cnp.float64_t   ,ndim=1] dzp,
        int n, bool dirichelet1=False, bool dirichelet2=False):
    r""" Assemble a load matrix from the coefficients of the differential equation.
    
    Cythonized assembly method, call as `assemble_load_matrix(w, dzp, n)`.
    
    For the mathematics/definitions of the :math:`w` term, see :ref:`FEM`.
    All arguments should be defined on the mid mesh, and the :math:`w` matrix should be one-dimensional floats
    
    Args:
        w: load-side coefficient
        dzp: the spacings between mesh nodes
        n: the size of the :math:`C` matrices (eg 1 for scalar equation, 6 for 6x6 kp equation)
        dirichelet1,dirichelet2 (bool): whether to treat boundary 1, 2 as Dirichelet (True, default) or Neumann.
    
    Returns:
        A load matrix in sparse CSC form
    """
    cdef:
        int i,r,nz,z, zmat
        double tmp
        list rinds, rdata
    nz=dzp.shape[0]+1
    lil=lil_matrix((n*(nz-dirichelet1-dirichelet2),n*(nz-dirichelet1-dirichelet2)),dtype='float')
    for z in range(dirichelet1,nz-dirichelet2):
        for i in range(n):
            zmat=z-dirichelet1
            r=zmat*n+i
            rinds=lil.rows[r]
            rdata=lil.data[r]

            #put in left
            if zmat>0:
                rinds+=[n*zmat-n+i]
                rdata+=[w[z-1]*dzp[z-1]/6]

            #put in diag
            rinds+=[n*zmat+i]
            tmp=0
            if z>0:
                tmp+=w[z-1]*dzp[z-1]/3
            if z<nz-1:
                tmp+=w[z  ]*dzp[z  ]/3
            rdata+=[tmp]

            #put in right
            if z<nz-1-dirichelet2:
                rinds+=[n*zmat+n+i]
                rdata+=[w[z  ]*dzp[z  ]/6]

    return lil.asformat('csc')

def fem_eigsh(stiffness_matrix,load_matrix,
              eigval_out,eigvec_out,n,
              dirichelet1=False,dirichelet2=False,*args,**kwargs):
    """ Solve the eigenvalue problem.

    For the mathematics/definitions of terms, see :ref:`FEM`.

    Args:
        stiffness_matrix: a stiffness matrix :math:`A` from :func:`pynitride.fem.assemble_stiffness_matrix`
        load_matrix: a load matrix :math:`M` from :func:`pynitride.fem.assemble_load_matrix`
        eigval_out: an array into which to fill the eigenvalues, should be shape (number of eigenvalues)
        eigvec_out: an array into which to fill the eigenvectors, should be shape (number of eigenvalues x n x len(zp))
        n: dimension of the differential equation
        dirichelet1,dirichelet2 (bool): whether to treat boundary 1, 2 as Dirichelet (True, default) or Neumann.
        *args,**kwargs: passed onto
            `scipy.sparse.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_
            along with the main arguments `A` (stiffness) and `M` (load)

    Returns:
        None
    """
    evec_slice=slice(dirichelet1,eigvec_out.shape[-1]-dirichelet2)
    eigval_out[:],eigvecs=eigsh(A=stiffness_matrix,M=load_matrix,*args,**kwargs)

    # Re-order the energies
    indarr=np.argsort(eigval_out)
    eigval_out[:]=eigval_out[indarr]

    # Reshape the eigenvectors to axes of (z, comp, eig)
    eigvecs.shape=(int(eigvecs.shape[0]/n),n,eigvecs.shape[1])

    # Then assign it to the output in axes of (eig, comp, z)
    if n==1: eigvec_out=np.expand_dims(eigvec_out,1)
    eigvec_out[:,:,evec_slice]=eigvecs.T[indarr,:,:]

    # Zeros at dirichelet boundaries
    if dirichelet1:
        eigvec_out[:,:,0]=0
    if dirichelet2:
        eigvec_out[:,:,-1]=0


def fem_solve(stiffness_matrix,load_matrix,load_vec,val_out,n,
              dirichelet1=False,dirichelet2=False,*args,**kwargs):
    """ Solve the linear matrix equation

    For the mathematics/definitions of terms, see :ref:`FEM`.

    Args:
        stiffness_matrix: a stiffness matrix :math:`A` from :func:`pynitride.fem.assemble_stiffness_matrix`
        load_matrix: a load matrix :math:`M` from :func:`pynitride.fem.assemble_load_matrix`
        load_vec: the load vector :math:`b`, should be shape (len(zp))
        val_out: an array into which to fill the solution, should be shape (len(zp))
        n: dimension of the differential equation
        dirichelet1,dirichelet2 (bool): whether to treat boundary 1, 2 as Dirichelet (True, default) or Neumann.
        *args,**kwargs: passed onto
            `scipy.sparse.spsolve <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html>`_
            along with the main arguments `A` (stiffness) and `M@b` (load)

    Returns:
        the solution (same shape as `load_vec`)
    """
    vslice=slice(dirichelet1*n,dirichelet1*n+stiffness_matrix.shape[0])
    load=load_matrix @ load_vec[vslice]
    if val_out is None:
        val_out=np.zeros_like(load_vec,dtype=stiffness_matrix.dtype)
    val_out[vslice]=spsolve(stiffness_matrix,load,*args,**kwargs)
    return val_out


def fem_get_error(stiffness_matrix,load_matrix,load_vec,test,err_out,n,
                  dirichelet1=False,dirichelet2=False,*args,**kwargs):
    """ Finds the signed error vector of the test solution.

    The error is :math:`b-M^{-1}Ax`, where :math:`x` is the supplied test solution.
    For the mathematics/definitions of terms, see :ref:`FEM`.

    Args:
        stiffness_matrix: a stiffness matrix :math:`A` from :func:`pynitride.fem.assemble_stiffness_matrix`
        load_matrix: a load matrix :math:`M` from :func:`pynitride.fem.assemble_load_matrix`
        load_vec: the load vector :math:`b`, should be shape (len(zp))
        test: the test solution vector :math:`x`, should be shape (len(zp))
        err_out: an array into which to fill the error, should be shape (len(zp))
        n: dimension of the differential equation
        dirichelet1,dirichelet2 (bool): whether to treat boundary 1, 2 as Dirichelet (True, default) or Neumann.
        *args,**kwargs: passed onto
            `scipy.sparse.spsolve <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html>`_
            along with the main arguments `M` and `A@x`

    Returns:
        the error (same shape as `test`)
    """
    vslice=slice(dirichelet1*n,dirichelet1*n+stiffness_matrix.shape[0])
    comp_load_vec=spsolve(load_matrix,stiffness_matrix @ test[vslice])
    if err_out is None:
        err_out=np.zeros_like(load_vec,dtype=stiffness_matrix.dtype)
    err_out[vslice]=load_vec[vslice]-comp_load_vec
    return err_out
