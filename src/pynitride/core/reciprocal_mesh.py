import numpy as np
from scipy.sparse import dok_matrix

pi=np.pi
from pynitride.core.machine import glob_store_attributes
from pynitride.core.maths import polar2cart, cart2polar, round_near
from scipy.interpolate import RectBivariateSpline, splrep, splev
from pynitride.visual.bands import white2red
import matplotlib.pyplot as plt

@glob_store_attributes('_functions')
class RMesh:
    def __init__(self, name=''):
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
        self.name=name
        """ A user-specified optional name"""

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
        """ Saves the contents of the reciprocal mesh to a Numpy .npz

        Args:
            filename: the path to save to
            keys: if given, only saves the specified list of keys
        """
        if keys is None:
            res=self._functions
        else:
            res={k:self[k] for k in keys}
        res['absk'  ]=self.absk
        res['absk1' ]=self.absk1
        if 'theta' in self:
            res['theta' ]=self.theta
            res['theta1']=self.theta1
        np.savez(filename,**res)
    def read(self,filename,keys=None):
        """ Reads the contents of the reciprocal mesh from a Numpy .npz

        Args:
            filename: the path to read from
            keys: if given, only reads the specified list of keys
        """
        with np.load(filename,allow_pickle=True) as data:
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
                    if keys is None or k in keys: 
                        self[k]=v

class RMesh1D(RMesh):
    """ A 1-D mesh of k-space.

    Note that when integrating over the mesh via :func:`RMesh.integrate`,
    it will behave as a 2D integral, ie the increased weighting of points
    at higher radius is accounted for intrinsically by this function."""
    def __init__(self,absk,bzarea=None,ival=None,name=''):
        super().__init__(name=name)

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
        if ival is not None:
            self._ival=ival
        else:
            self._ival=np.min(dabsk)/10
        self._k2i={k:i for i,k in enumerate(round_near(self.absk1,self._ival))}

        # Define the other parts of the parameterization
        self.theta=np.zeros_like(self.absk)
        self.kx=self.absk
        self.ky=np.zeros_like(self.absk)
        self.N=len(self.absk)


    @classmethod
    def regular(cls,kmax,numabsk,abskshift=0,name=''):
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
        return self._k2i[round_near(absk,self._ival)]

    def absk_subrmesh(self,indices):
        """ Forms an Rmesh1D which is subset of this one, keeping only the specified indices

        Args:
            indices: list of indices into the RMesh which should be kept
        """
        assert not isinstance(indices,slice), "indices should be a list/array"
        absk=self.absk1[indices]
        
        sub=RMesh1D(absk,bzarea=self.bzarea,ival=self._ival)

        for key,val in self._functions.items():
            try:
                sub._functions[key]=val[indices]
            except:
                sub._functions[key]=[val[iq] for iq in indices]
        return sub

    def interpolator(self,func):
        """ Returns a 1-D spline interpolation of func

        The returned function is called like
        `interp(absk, theta=0, dabsk=0, bounds_check=True)`
        where `absk` is the radial coordinate,
        `theta` is ignored (present for compatibility with RMesh2D code),
        'dabsk` specifies order of radial derivative,
        and `bounds_check` specifies whether to throw out-of-bounds errors

        Args:
            func: the function to interpolate

        """
 
        # Get the spline interpolation using splrep/splev instead of
        # interp1d because it can also provide derivatives
        tck, fp, ier, msg=splrep(self.absk1,func,full_output=True)
        assert ier<=0, msg

        def interp(absk, theta=0, grid=False, dabsk=0, bounds_check=True):
            if grid:
                raise NotImplementedError("The grid argument to RMesh1D.interpolator.interp() was added only to match"\
                        " the signature of the similar function for RMesh2D, and hasn't been implemented yet,"\
                        " so please only supply grid=False")
            # for out of bounds, ext=0 extrapolates, ext=2 raises an error
            return splev(absk,tck,der=dabsk,ext=bounds_check*2)
        return interp


class RMesh2D_Polar(RMesh):
    """ A 2-D mesh of k-space."""
    def __init__(self,absk,theta,d=1,bzarea=None,name=''):
        super().__init__(name=name)

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
    def regular(cls,kmax,numabsk,numtheta,include_kzero=True,align_theta=False,d=1,name=''):
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
        return cls.regular_theta(absk,numtheta,align_theta,d,name=name)

    @classmethod
    def regular_theta(cls,absk,numtheta,align_theta=False,d=1,name=''):
        r""" Generate a 2D grid with regular `theta` spacing.

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
        return RMesh2D_Polar(absk,theta,d,bzarea=None,name=name)


    def conv2grid(self,arr):
        r""" Convert a 1D array to a 2D grid consistent with the mesh ordering.

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
        arr=arr.T
        arr=np.reshape(arr,[np.prod(arr.shape[:2])]+list(arr.shape[2:]))
        if self.absk1[0]==0:
            return arr[self.numtheta-1:]
        else:
            return arr

    def interpolator(self,func):
        """ Returns a 2-D spline interpolation of func

        The returned function is called like
        `interp(absk, theta, grid=False, dabsk=0, dtheta=0, bounds_check=True)`
        where `absk`, `theta` are the coordinates,
        `grid` is as in :py:class:`scipy.interpolate.BivariateSpline`,
        `dabsk`, `dtheta` specify order of respective derivatives,
        and `bounds_check` specifies whether to throw out-of-bounds errors

        Args:
            func: the function to interpolate

        """
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
            if grid:
                theta=np.unwrap(theta)

            return rbvs(absk,theta,grid=grid, dx=dabsk, dy=dtheta)
        return interp

    def absk_subrmesh(self,abskstart=1,abskstop=-1,name=None):
        """ Produce an RMesh which is a contiguous subset of this one in absk

        Args:
            abskstart: the absk index to start on
            abskstop: the absk index to stop before
            name: the name for the new RMesh

        """
        abskslice=slice(abskstart,abskstop)
        absk=self.absk1[abskslice]
        theta=self.theta1
        
        if name is None: name=self.name
        sub=RMesh2D_Polar(absk,theta,d=self.d,bzarea=self.bzarea,name=name)

        start=list(self.absk).index(absk[0])
        stop=self.N-list(self.absk[::-1]).index(absk[-1])
        for key,val in self._functions.items():
            sub._functions[key]=val[start:stop]
        sub.supermesh=self
        return sub

    def partial_indices_to_index(self,iabsk,itheta):
        """ Given an index into `absk` and into `theta`, returns an index into the RMesh"""
        return self._ikit2i[itheta,iabsk]
    def index_to_partial_indices(self,i):
        """ Given an index into the RMesh, returns indices into `absk` and into `theta`"""
        return self._i2ik[i],self._i2it[i]

    def ikx(self,sign=False):
        """ Returns the indices into the RMesh corresponding to a slice along `ky=0`.

        Args:
            sign: if True, include only non-negative values of kx
        """
        ikx=np.argsort(self.kx)
        ikx=ikx[np.isclose(self.ky[ikx],0,atol=1e-10)]
        if sign:
            ikx=ikx[sign*self.kx[ikx]>=0]
        return ikx
    def iky(self,sign=False):
        """ Returns the indices into the RMesh corresponding to a slice along `kx=0`.

        Args:
            sign: if True, include only non-negative values of ky
        """
        iky=np.argsort(self.ky)
        iky=iky[np.isclose(self.kx[iky],0,atol=1e-10)]
        if sign:
            iky=iky[sign*self.ky[iky]>=0]
        return iky

    def theta_diff_mat(self, for_pre_integrated_values=False):
        """ Returns a finite-differences matrix for differentiating a function in the theta direction.

        Note: the thing that's differentiated is df/k*dtheta, not d[f*Omega]/k*dtheta
        If you want a matrix that applies to f*Omega but still returns df/dtheta, supply for_pre_integrated_values=True
        :return: a (sparse) matrix M such that df/k*dtheta is approximated by M*f
        """
        # There's probably a clever vectorized way to do this, but this function is so far
        # only used in the final solve_BTE, not inside any important loops so...
        tdiff_mat = dok_matrix((self.N, self.N))
        fdt = np.mod(np.roll(self.theta1, -1) - self.theta1, 2 * np.pi)
        rdt = np.mod(self.theta1 - np.roll(self.theta1, +1), 2 * np.pi)
        for ik in range(self.numabsk):
            if self.absk[self.partial_indices_to_index(ik, 0)] != 0:
                for it in range(self.numtheta):
                    i_this = self.partial_indices_to_index(ik, it)
                    i_next = self.partial_indices_to_index(ik, (it + 1) % self.numtheta)
                    i_prev = self.partial_indices_to_index(ik, (it - 1) % self.numtheta)
                    # If the array this matrix will get multiplied by is f*Omega
                    if for_pre_integrated_values:
                        tdiff_mat[i_this, i_next] = +self.Omega[i_this] / self.Omega[i_next] / (
                                   2 * self.absk[i_this] * fdt[it])
                        tdiff_mat[i_this, i_prev] = -self.Omega[i_this] / self.Omega[i_prev] / (
                                   2 * self.absk[i_this] * rdt[it])
                    else:
                    # If the array this matrix will get multiplied by is f
                        tdiff_mat[i_this, i_next] = + 2 * self.absk[i_this] * fdt[it]
                        tdiff_mat[i_this, i_prev] = - 2 * self.absk[i_this] * rdt[it]
        return tdiff_mat

    def absk_diff_mat(self):
        raise NotImplementedError("Haven't implemented finite difference matrix in absk direction")


    def show_func(self,func,style='balanced',points=True, lines=True,
            cax=None,vmax=None,numloc=1000,label=None):
        """ Visualize func on a rasterized 2D colormesh plot.

        Args:
            func: the function (as a 1-D array of the same shape as the RMesh variables)
            style: 'balanced' will give a red-blue plot with the color-scale set symmetrically, while
                'positive' will give a white-to-red plot where white is the zero
            points: whether to plot a point marker at the center of each mesh element
            lines: whether to plot the boundary lines of each mesh element
            cax: optional axes for the colorscale (if not supplied, will be created)
            vmax: for the 'positive' style, a vmax can be specified (otherwise chosen automatically)
            numloc: number of points each along kx and ky to use when forming the colormesh

        """

        kx=np.linspace(-self.kmax,self.kmax,numloc)
        ky=np.linspace(-self.kmax,self.kmax,numloc)
        
        KX,KY=np.meshgrid(kx,ky)
        ABSK,THETA=cart2polar(KX,KY)
        
        iabsk=np.digitize(ABSK,self.abskbinu)
        valid=iabsk<self.numabsk
        iabsk[~valid]=0
        itheta=np.mod(np.digitize(THETA,self.thetabinu),self.numtheta)
        
        i=self.partial_indices_to_index(iabsk,itheta)
        F=func[i]
        F[~valid]=np.nan
        
        if style=='balanced':
            vmin=np.nanmin(F)
            vmax=np.nanmax(F)
            vmin,vmax=np.array([-1,1])*np.max(np.abs([vmin,vmax]))
            cmap='seismic'
            dontplot=False
        if style=='positive':
            vmin=0
            if vmax is None: 
                vmax=np.nanmax(F)
            cmap=white2red
            dontplot=(vmax==0)
        if not dontplot:
            plt.pcolormesh(KX,KY,F,vmin=vmin,vmax=vmax,cmap=cmap,rasterized=True)

        if points:
            plt.plot(self.kx,self.ky,'o',markersize=3)

        plt.axis('square')
        plt.xlim(-self.kmax,self.kmax)
        plt.ylim(-self.kmax,self.kmax)

        if not dontplot:
            cb=plt.colorbar(cax=cax)
            if label:
                cb.set_label(label)

        if lines:
            for t in self.thetabinl:
                plt.plot([0,10*np.cos(t)],[0,10*np.sin(t)],
                        color='gray',linewidth=.5,rasterized=True)
            for k in self.abskbinl:
                plt.plot(*polar2cart(k,np.linspace(0,2*pi,endpoint=True)),
                        color='gray',linewidth=.5,rasterized=True)
        plt.xlabel("$k_x$ [1/nm]")
        plt.ylabel("$k_y$ [1/nm]")
