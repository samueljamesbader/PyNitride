import numpy as np
pi=np.pi
from pynitride.machine import glob_store_attributes
from pynitride.maths import polar2cart, cart2polar
from scipy.interpolate import RectBivariateSpline, splrep, splev
from pynitride.visual import white2red
import matplotlib.pyplot as plt

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
        self.d=None
        """ The grid covers 1/d of the angular k-space"""
        self.kmax=None
        """ The maximum value of :math:`|k|`"""

        # Place to store functions defined on this mesh
        self._functions={}

    # Store/retrieve reciprocal mesh functions via [...] syntax
    def __setitem__(self, key, value):
        self._functions[key]=value
    def __getitem__(self, key):
        return self._functions[key]
    def __contains__(self, key):
        return (key in self._functions)
    def __delitem__(self, key):
        del self._functions[key]


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
        return self.d*np.sum(integrand.T*self.Omega,axis=-1).T

    def save(self,filename,keys=None):
        if keys is None:
            res=self._functions
        else:
            res={k:self[k] for k in keys}
        res['absk']=self.absk
        res['theta']=self.theta
        np.savez(filename,**res)
    def read(self,filename):
        with np.load(filename) as data:
            for k,v in data.items():
                # Check grid coordinates
                if k=='absk':
                    assert np.allclose(v,self.absk),\
                        "Loaded rmesh does not match current."
                elif k=='theta':
                    assert np.allclose(v,self.theta),\
                        "Loaded rmesh does not match current."
                # Store functions
                else:
                    self[k]=v

class RMesh1D(RMesh):
    """ A 1-D mesh of k-space.

    Note that when integrating over the mesh via :func:`RMesh.integrate`,
    it will behave as a 2D integral, ie the increased weighting of points
    at higher radius is accounted for intrinsically by this function."""
    def __init__(self,absk,bzarea=None):
        super().__init__()

        # k and dk
        self.absk1=self.absk=np.sort(absk)
        self.kmax=self.absk1[-1]
        dabsk=self.absk1[1:]-self.absk1[:-1]
        self.bzarea=bzarea

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
        self.d=1

        # Mapping from absk1 values to their indices, used by exact_to_index
        self._k2i={k:i for i,k in enumerate(np.round(self.absk1,self._exactdig))}

        # Define the other parts of the parameterization
        self.theta=np.zeros_like(self.absk)
        self.kx=self.absk
        self.ky=np.zeros_like(self.absk)
        self.N=len(self.absk)

    _exactdig=7

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

    def absk_subrmesh(self,indices):
        assert not isinstance(indices,slice), "indices should be a list/array"
        absk=self.absk1[indices]
        
        sub=RMesh1D(absk,bzarea=self.bzarea)

        for key,val in self._functions.items():
            sub._functions[key]=val[indices]
        return sub

    def interpolator(self,func):
 
        # Get the spline interpolation using splrep/splev instead of
        # interp1d because it can also provide derivatives
        tck, fp, ier, msg=splrep(self.absk1,func,full_output=True)
        assert ier<=0, msg

        def interp(absk, dabsk=0, bounds_check=True):
            # for out of bounds, ext=0 extrapolates, ext=2 raises an error
            return splev(absk,tck,der=dabsk,ext=bounds_check*2)
        return interp


class RMesh2D_Polar(RMesh):
    """ A 2-D mesh of k-space."""
    def __init__(self,absk,theta,d=1,bzarea=None):
        super().__init__()

        self.absk1=np.sort(absk)
        self.kmax=self.absk1[-1]
        self.theta1=np.sort(theta)
        self.numabsk=len(self.absk1)
        self.numtheta=len(self.theta1)
        self.d=d
        self.bzarea=bzarea

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
        ABSK,THETA=np.meshgrid(self.absk1,self.theta1)
        self.absk=self.conv2flat(ABSK)
        self.theta=self.conv2flat(THETA)
        if self.absk1[0]==0:
            self.theta[0]=0
        self.kx,self.ky=polar2cart(self.absk,self.theta)
        self.N=len(self.absk)

        # Area to use for each when integrating
        Oabsk=(self.abskbinu**2-self.abskbinl**2)/2
        self.dtheta=(dthetal+dthetar)/2
        OABSK,OTHETA=np.meshgrid(Oabsk,self.dtheta)
        self.Omega=self.conv2flat(OABSK*OTHETA)
        if self.absk1[0]==0:
            self.Omega[0]*=self.numtheta

        # Mapping from absk1 values to their indices
        self._exactdig=7
        self._k2i={k:i for i,k in enumerate(np.round(self.absk1,self._exactdig))}
        self._ikit2i=self.conv2grid(np.arange(self.N))
        self._i2ik=self.conv2flat(np.meshgrid(np.arange(len(self.absk1)),np.arange(len(self.theta1)))[0])
        self._i2it=self.conv2flat(np.meshgrid(np.arange(len(self.absk1)),np.arange(len(self.theta1)))[1])

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
        return np.reshape(arr,[self.numabsk,self.numtheta]+list(arr.shape[1:])).T

    def conv2flat(self,arr):
        """ Convert a 2D grid to a 1D array consistent with the mesh conventions.

        Args:
            arr: doesn't need to be 2D per se, but the first two dimensions
                must be numabsk and numtheta long respectively.  Other
                dimensions just along for the ride.
        Returns:
            an (numstates) 1D array + other dimensions
        """
        arr=np.reshape(arr.T,[np.prod(arr.shape[:2])]+list(arr.shape[2:]))
        if self.absk1[0]==0:
            return arr[self.numtheta-1:]
        else:
            return arr

    #def exact_to_index(self, absk):
    #    """ Returns the index into the absk1 array for a given k-point"""
    #    return self._k2i[np.round(absk,self._exactdig)]

    def interpolator(self,func):
        d=self.d
        assert d==1, "Interpolation only provided for full k-space RMesh"

        # Which if any side of theta is at the mod boundary
        atedge0=np.isclose(self.theta1[ 0],-pi/d)
        atedge1=np.isclose(self.theta1[-1],+pi/d)

        # Extend theta with two points beyond the integration boundary on both sides
        theta=np.concatenate([
            self.theta1[(-3+atedge0):]-2*pi/d,
            self.theta1,
            self.theta1[:(+3-atedge1)]+2*pi/d])

        # Join in the energy
        fmain=self.conv2grid(func)
        f=np.vstack([fmain[(-3+atedge0):,:],fmain,fmain[:(+3-atedge1),:]])
        rbvs=RectBivariateSpline(self.absk1,theta,f.T,
            bbox=[0,self.absk1[-1],theta[0],theta[-1]])

        def interp(absk,theta,grid=False, dabsk=0, dtheta=0, bounds_check=True):
            if bounds_check:
                assert np.all(absk<=self.absk1[-1]), "Out of interpolation range"
            theta=np.mod(np.mod(theta,2*pi/d)+pi/d,2*pi/d)-pi/d
            return rbvs(absk,theta,grid=grid, dx=dabsk, dy=dtheta)
        return interp

    def absk_subrmesh(self,abskstart=1,abskstop=-1):
        abskslice=slice(abskstart,abskstop)
        absk=self.absk1[abskslice]
        theta=self.theta1
        
        sub=RMesh2D_Polar(absk,theta,d=self.d,bzarea=self.bzarea)

        start=list(self.absk).index(absk[0])
        stop=self.N-list(self.absk[::-1]).index(absk[-1])
        for key,val in self._functions.items():
            sub._functions[key]=val[start:stop]
        sub.supermesh=self
        return sub

    def partial_indices_to_index(self,iabsk,itheta):
        return self._ikit2i[itheta,iabsk]
    def index_to_partial_indices(self,i):
        return self._i2ik[i],self._i2it[i]

    def show_func(self,func,style='balanced',points=True):

        kx=np.linspace(-self.kmax,self.kmax,1000)
        ky=np.linspace(-self.kmax,self.kmax,1000)
        
        KX,KY=np.meshgrid(kx,ky)
        ABSK,THETA=cart2polar(KX,KY)
        
        iabsk=np.digitize(ABSK,self.abskbinu)
        valid=iabsk<self.numabsk
        iabsk[~valid]=0
        itheta=np.mod(np.digitize(THETA,self.thetabinu),self.numtheta)
        
        i=self.partial_indices_to_index(iabsk,itheta)
        F=func[i]
        F[~valid]=np.NaN
        
        if style=='balanced':
            vmin=np.nanmin(F)
            vmax=np.nanmax(F)
            vmin,vmax=np.array([-1,1])*np.max(np.abs([vmin,vmax]))
            cmap='seismic'
        if style=='positive':
            vmin=0
            vmax=np.nanmax(F)
            cmap=white2red
        plt.pcolormesh(KX,KY,F,vmin=vmin,vmax=vmax,cmap=cmap)

        if points:
            plt.plot(self.kx,self.ky,'o',markersize=3)

        plt.axis('square')
        plt.xlim(-self.kmax,self.kmax)
        plt.ylim(-self.kmax,self.kmax)

        plt.colorbar()

        for t in self.thetabinl:
            plt.plot([0,10*np.cos(t)],[0,10*np.sin(t)],color='gray',linewidth=.5)
        for k in self.abskbinl:
            plt.plot(*polar2cart(k,np.linspace(0,2*pi,endpoint=True)),color='gray',linewidth=.5)
        plt.xlabel("$k_x$ [1/nm]")
        plt.ylabel("$k_y$ [1/nm]")
