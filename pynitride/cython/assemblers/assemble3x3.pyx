cimport cython
cimport numpy as cnp
import numpy as np
from scipy.sparse import lil_matrix
cnp.import_array()
from cpython cimport bool
from libc.math cimport sqrt


DEF n=3
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef assemble3x3(
        cnp.ndarray[cnp.float64_t,ndim=3] C0, cnp.ndarray[cnp.float64_t,ndim=3] Cl,
        cnp.ndarray[cnp.float64_t,ndim=3] Cr, cnp.ndarray[cnp.float64_t,ndim=3] C2,
        cnp.ndarray[cnp.float64_t   ,ndim=1] dzm,cnp.ndarray[cnp.float64_t   ,ndim=1] dzp,
        float bctop, float bcbottom):
    #assert bctop=="Free"
    #assert bcbottom=="Fixed"
    cdef:
        int i,j,r,nz,z
        list rinds, rdata
        cnp.complex128_t[n] tmp
    nz=C0.shape[2]
    lil=lil_matrix((n*nz,n*nz),dtype='complex')
    for z in range(nz):
        for i in range(n):
            r=z*n+i
            rinds=lil.rows[r]
            rdata=lil.data[r]

            # Put in left
            if z>0:
                rinds+=range(n*z-n,n*z)
                for j in range(n):
                    tmp[j]= \
                        -C2[i,j,z-1]/dzp[z-1]/sqrt(dzm[z]*dzm[z-1]) \
                        +.5j*(Cl[i,j,z]+Cr[i,j,z-1])/sqrt(dzm[z]*dzm[z-1])
                rdata+=[t for t in tmp]

            # Put in diag
            rinds+=range(n*z,n*z+n)
            for j in range(n):
                tmp[j]= \
                    C0[i,j,z]+ \
                    (C2[i,j,z-1]/dzp[z-1]/dzm[z] if z>0    else 0) + \
                    (C2[i,j,z  ]/dzp[z  ]/dzm[z] if z<nz-1 else 0) + \
                    ((1j/dzp[z  ]*(Cl[i,j,z]-Cr[i,j,z])-   bctop*(i==j)/dzp[z  ]) if z==0    else 0)+ \
                    ((1j/dzp[z-1]*(Cl[i,j,z]-Cr[i,j,z])-bcbottom*(i==j)/dzp[z-1]) if z==nz-1 else 0)
            rdata+=[t for t in tmp]

            # Put in right
            if z<nz-1:
                rinds+=range(n*z+n,n*z+2*n)
                for j in range(n):
                    tmp[j]= \
                        -C2[i,j,z]/dzp[z]/sqrt(dzm[z]*dzm[z+1]) \
                        -.5j*(Cl[i,j,z]+Cr[i,j,z+1])/sqrt(dzm[z]*dzm[z+1])
                rdata+=[t for t in tmp]
    return lil.asformat('csc')
