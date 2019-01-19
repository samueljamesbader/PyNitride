import numpy as np
pi=np.pi
from pynitride.machine import glob_store_attributes
from pynitride.maths import polar2cart


class KMesh2D:
    def __init__(self,kx,ky):
        self.kx1=kx
        self.kx1p=kx[kx>=0]
        self.ky1=ky
        self.ky1p=ky[ky>=0]
        self.lkx=len(self.kx1)
        self.lky=len(self.ky1)
        self.KX,self.KY=np.meshgrid(kx,ky)
        self.KT=np.stack([self.KX,self.KY],axis=2)
        self.kt=self.conv2flat(self.KT)
        self.kx=self.kt[:,0]
        self.ky=self.kt[:,1]
        if len(kx)>1:
            self.dkx=kx[1]-kx[0]
        if len(ky)>1:
            self.dky=ky[1]-ky[0]

    def conv2grid(self,arr):
        return np.reshape(arr,[self.lky,self.lkx]+list(arr.shape[1:]))
    def conv2flat(self,arr):
        return np.reshape(arr,[np.prod(arr.shape[:2])]+list(arr.shape[2:]))
    def intflat(self,arr):
        ig= self.conv2grid(arr).T
        return np.trapz(np.trapz(ig,dx=self.dky),dx=self.dkx).T
    def along(self,arr,dir='x',input='guess',onesided=True):
        assert input=='flat' or (input=='guess' and arr.shape[0]==self.lkx*self.lky)
        if dir=='x':
            iy=np.argmax(self.ky1==0)
            assert (self.ky1[iy]==0), "0 not in ky"
            ix = np.argmax(self.kx1>=0) if onesided else 0
            return arr[(iy*self.lkx+ix):(iy*self.lkx+self.lkx)]
        elif dir=='y':
            ix=np.argmax(self.kx1==0)
            assert (self.kx1[ix]==0), "0 not in kx"
            iy=np.argmax(self.ky1>=0) if onesided else 0
            return arr[(ix+iy*self.lkx)::self.lkx]

@glob_store_attributes('_functions')
class RMesh:
    def __init__(self):
        """ Superclass for all the reciprocal meshes"""
        self.absk=None
        """ The absk values of the grid as a 1-D array"""
        self.theta=None
        """ The `theta` values of the grid as a 1-D array"""
        self.kx=None
        """ The `kx` values of the grid as a 1-D array"""
        self.ky=None
        """ The `ky` values of the grid as a 1-D array"""
        self.Omega=None
        """ The integration area element assigned to each kpoint as a 1-D array"""
        self.N=None
        """ The number of points in the grid"""

        self._functions={}

    def __setitem__(self, key, value):
        self._functions[key]=value
    def __getitem__(self, key):
        return self._functions[key]


    def integrate(self,integrand):
        r""" Integrates the integrand over the RMesh.
        For instance, if the integrand is 1, you should get the area of the k-space.
        For a regular mesh with no shift, that would be :math:`\pi\times kmax^2`
        For a mesh of the full Brillouin zone, that would be the full `bzarea`.

        Note that this integral is *just an integral over k-space*, ie does not magically include the
        :math:`1/4\pi^2` pre-factor which appears when converting a sum over k-space to an integral over k-space.

        Args:
            integrand: a 1-D array of length `RMesh.N`
        """
        return np.sum(integrand.T*self.Omega,axis=-1).T

    def save(self,filename,keys=None):
        if keys is None:
            res=self._functions
        else:
            res={k:self[k] for k in keys}
        np.savez_compressed(filename,**res)
    def read(self,filename):
        with np.load(filename) as data:
            for k,v in data.items():
                self[k]=v

class RMesh1D(RMesh):
    """ A 1-D mesh of k-space.

    Note that when integrating over the mesh via :func:`RMesh.integrate`, it will behave as a 2D integral, ie
    the increased weighting of points at higher radius is already accounted for."""
    def __init__(self,absk,bzarea=None):
        super().__init__()

        # k and dk
        self.absk1=self.absk=absk
        dabsk=self.absk1[1:]-self.absk1[:-1]

        # lower bound of each bin, including zero for the first bin
        self._abskbinl=np.concatenate([[0],self.absk1[:-1]+dabsk/2])

        # upper bound of each bin, including the last binpoint as its own upper bound
        self._abskbinu=np.concatenate([self.absk1[:-1]+dabsk/2,[self.absk1[-1]]])

        # BUT if bzarea is provided, then expand the upper bound of the last bin such that
        # total brillouin zone area is represented as if the BZ were a circle.
        if bzarea is not None:
            self._abskbinu[-1]=np.sqrt(bzarea/pi)

        # Area to use for each when integrating
        self.Omega=pi*(self._abskbinu**2-self._abskbinl**2)

        # Mapping from absk1 values to their indices, used by exact_to_index
        self._exactdig=7
        self._k2i={k:i for i,k in enumerate(np.round(self.absk1,self._exactdig))}

        # Define the other parts of the parameterization
        self.theta=np.zeros_like(self.absk)
        self.kx=self.absk
        self.ky=np.zeros_like(self.absk)
        self.N=len(self.absk)

    @classmethod
    def regular(cls,kmax,numabsk,abskshift=0):
        """ Generate a uniform 1D grid.

        The k-points go from 0 to kmax inclusive, but with an optional shift.
        The endpoints of the mesh are treated as endpoints of integration.

        Args:
            kmax (float): the largest k-point will be kmax+abskshift
            numabsk (int): number of k-points to have
            abskshift (float): shift all k-points from starting at zero

        Returns:
            a :class:`RMesh1D`
        """
        return cls(np.linspace(0,kmax,num=numabsk)+abskshift,bzarea=None)

    def exact_to_index(self, absk):
        """ Returns the index into the absk1 array for a given k-point"""
        return self._k2i[np.round(absk,self._exactdig)]


class RMesh2D_Polar(RMesh):
    """ A 2-D mesh of k-space."""
    def __init__(self,absk,theta,d=1,bzarea=None):
        super().__init__()

        self.absk1=absk
        self.theta1=theta
        self.numabsk=len(self.absk1)
        self.numtheta=len(self.theta1)

        ###
        # k bin boundaries

        # differences between absk points
        dabsk=self.absk1[1:]-self.absk1[:-1]

        # lower bound of each bin, including zero for the first bin
        self.abskbinl=np.concatenate([[0],self.absk1[:-1]+dabsk/2])

        # upper bound of each bin, including the last binpoint as its own upper bound
        self.abskbinu=np.concatenate([self.absk1[:-1]+dabsk/2,[self.absk1[-1]]])

        # BUT if bzarea is provided, then expand the upper bound of the last bin such that
        # total brillouin zone area is represented as if the BZ were a circle.
        if bzarea is not None:
            self.abskbinu[-1]=np.sqrt(bzarea/pi)
        ###

        ###
        # theta bin boundaries

        # differences between theta points
        inner_dtheta=self.theta1[1:]-self.theta1[:-1]
        extreme_dtheta= self.theta1[0]-(self.theta1[-1]-2*pi/d)

        # left differences and right differences
        dthetal=np.concatenate([[extreme_dtheta],inner_dtheta])
        dthetar=np.concatenate([inner_dtheta,[extreme_dtheta]])

        # lower and upper bound of each bin
        self.thetabinl,self.thetabinu=self.theta1-dthetal/2 , self.theta1+dthetar/2
        ###

        # Form the actual grid
        THETA,ABSK=np.meshgrid(self.theta1,self.absk1)
        self.absk=self.conv2flat(ABSK)
        self.theta=self.conv2flat(THETA)
        self.theta[0]=0
        self.kx,self.ky=polar2cart(self.absk,self.theta)
        self.N=len(self.absk)

        # Area to use for each when integrating
        Oabsk=(self.abskbinu**2-self.abskbinl**2)/2
        Otheta=(dthetal+dthetar)/2
        OTHETA,OABSK=np.meshgrid(Otheta,Oabsk)
        self.Omega=d*self.conv2flat(OABSK*OTHETA)
        if self.absk1[0]==0:
            self.Omega[0]*=self.numtheta

        # Mapping from absk1 values to their indices
        self._exactdig=7
        self._k2i={k:i for i,k in enumerate(np.round(self.absk1,self._exactdig))}

    @classmethod
    def regular(cls,kmax,numabsk,numtheta,include_kzero=True,align_theta=False,d=1):
        """ Generate a regular 2D grid.

        The absk1-points go from `kmax`/`numabsk` to `kmax` or `0` to `kmax` depending on `include_kzero`.
        Regardless, `0` and the `kmax` are treated as endpoints of integration.

        This function is just a thin wrapper around :func:`RMesh.uniform_theta`.
        See there for more arguments and details.

        Args:
            kmax (float): the largest k-point will be kmax
            numabsk (int): number of k-points to have


        Returns:
            a :class:`RMesh1D`
        """
        if include_kzero:
            absk=np.linspace(0,kmax,num=numabsk)
        else:
            absk=np.linspace(kmax/numabsk,kmax,num=numabsk)
        return cls.regular_theta(absk,numtheta,align_theta,d)

    @classmethod
    def regular_theta(cls,absk,numtheta,align_theta=False,d=1):
        """ Generate a 2D grid with regular `theta` spacing.

        Regardless of whether `absk` includes a point at `0`,
        `0` and the max(`absk`) are treated as endpoints of integration.

        The integration endpoints of `theta` are :math:`-pi/d` and :math:`+\pi/d`.
        By choosing `d` and `shift_theta`, one can choose any proper fraction of k-space to cover.

        Args:
            absk: the ordered array of k-points
            numtheta (int): number of theta points to have
            include_zero (bool): whether to have a point at 0
            align_theta (bool): If `True` there is a `theta` at 0.
                If `False`, then the points will be shifted so a midpoint between thetas is at 0.
            d (int): what fraction of the k-space to cover angularly
                (d=1 is the whole k-space, d=2 is half, etc.)

        Returns:
            a :class:`RMesh1D`
        """
        if (align_theta is False and (numtheta % 2 == 1)) or (align_theta is True and (numtheta % 2 == 0)):
            shift_theta=0
        else:
            shift_theta=pi/(numtheta*d)#+align_theta

        theta=np.linspace(-pi/d,pi/d,num=numtheta,endpoint=False)+shift_theta
        return RMesh2D_Polar(absk,theta,d,bzarea=None)


    def conv2grid(self,arr):
        """ Convert a 1D array to a 2D grid consistent with the mesh ordering.

        Args:
            arr: doesn't need to be 1D per se, but the first dimension must
                be numstates long.  Other dimensions just along for the ride.
        Returns:
            a (numabsk,numtheta) 2D array + other dimensions.
        """
        if self.absk1[0]==0:
            arr=np.concatenate([[arr[0]]*(self.numtheta-1),arr])
        return np.reshape(arr,[self.numtheta,self.numabsk]+list(arr.shape[1:]))

    def conv2flat(self,arr):
        """ Convert a 2D grid to a 1D array consistent with the mesh conventions.

        Args:
            arr: doesn't need to be 2D per se, but the first two dimensions
                must be numabsk and numtheta long respectively.  Other
                dimensions just along for the ride.
        Returns:
            an (numstates) 1D array + other dimensions
        """
        arr=np.reshape(arr,[np.prod(arr.shape[:2])]+list(arr.shape[2:]))
        if self.absk1[0]==0:
            return arr[self.numtheta-1:]
        else:
            return arr

    def exact_to_index(self, absk):
        """ Returns the index into the absk1 array for a given k-point"""
        return self._k2i[np.round(absk,self._exactdig)]
