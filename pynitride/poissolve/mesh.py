# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:15:41 2017

@author: sam
"""

import matplotlib.pyplot as mpl
import numpy as np
from matplotlib import pyplot as mpl
from scipy.interpolate import interp1d
import pickle
from math import gcd,ceil
from functools import reduce

from pynitride.paramdb import Material, ParamDB


class Layer():
    def __init__(self, name, matname, thickness, pmdb=ParamDB()):
        self._name = name
        self._matname = matname
        self._mat = Material(matname,pmdb=pmdb)
        self._thickness = thickness

    @property
    def name(self):
        return self._name

    @property
    def material(self):
        return self._mat

    @property
    def thickness(self):
        return self._thickness

    # if not found and default=... is passed, will return that instead of error
    def get(self, key, default=None):
        return self._mat(key,default=default)

    def __getitem__(self, key):
        return self._mat[key]


class EpiStack():
    def __init__(self,*args,surface=None,pmdb=ParamDB()):
        if isinstance(args[0],Layer):
            self._layers=args
        else:
            self._layers=[Layer(l[0], l[1], l[2],pmdb=pmdb) if len(l) == 3 else Layer(l[0], l[0], l[1],pmdb=pmdb) for l in args]
        self._surface=surface
        self._pmdb=pmdb

    @property
    def layers(self):
        return self._layers

    def __iter__(self):
        return iter(self._layers)

    def __getitem__(self, item):
        return self._layers[item]

    @property
    def materials(self):
        return set(l.material for l in self._layers)

    @property
    def surface(self):
        return self._surface

class Mesh():
    r""" Generates and manages a dual, potentially non-uniform mesh and functions defined on it.

    See :ref:`Meshing Scheme <mesh>` for a discussion of the defintion and properties of the mesh.

    The algorithm to create the mesh is to step through the regions one-by-one and build the mesh point-by-point.
    At a point :math:`z`, the mesh will be extended by adding a maximal :math:`dz` which satisfies all the refinement
    criteria (as a simplification, the criteria are evaluated at :math:`z`, but not :math:`z+dz`).  When the mesh being
    constructed passes an interface, the entire mesh built since the last interface is shrunk uniformly so that the
    most recent mesh point matches that interface.

    Refinement criteria are given as a max spacing :math:`dz^p_0` near a refinement point :math:`z_0`, and an
    exponential growth rate for that :math:`dz`.  This makes it easy to evaluate at any :math:`z` what is the maximal
    allowed :math:`dz`.  As mentioned, when the mesh is built, the criteria are evaluated at the most recent point z,
    not at :math:`z+dz`.  So it is possible that :math:`dz` may be slightly larger than the refinement criteria if the
    limiting refinement is to the right.  If the mesh is sufficiently dense, the refinement criteria will be
    approximately satisfied. But, because of this detail, note that these criteria are targets (which will be nearly
    hit), not guarantees.

    :param stack: the :py:class:`~pynitride.poissolve.mesh.EpiStack` representing the device
    :param max_dz: (number) the maximum mesh spacing allowed globally.
    :param refinements: a list of spots where the mesh should be refined.
        Each element is a triple :math:`(z^p_0,dz^p_0,r)`,
        where :math:`z_0` is the location that should be refined,
        :math:`dz^p_0` is the target mesh spacing in the region of :math:`z_0`,
        and :math:`r` is the target rate at which the mesh spacing is allowed to exponentially increase
        moving away from :math:`z_0`. ie each refinement enforces a constraint of the form
        :math:`dz \lesssim dz^p_0  r^{|z-z_0|}`.
        Note that, when creating a uniform mesh, the only effect of this argument is to reduce ``max_dz`` if a
        refinement with `dz^p_0` tighter than ``max_dz`` is included.
    """

    def __init__(self, stack, max_dz, refinements=[], uniform=False):

        # Make a uniform mesh
        if uniform:

            # Get the maximum spacing from the tightest of max_dz and the refinements
            max_dz=min([max_dz]+[r[1] for r in refinements])

            # Quick util to round and convert to an integer type
            rint=lambda x: int(round(x))

            # Get the gcd of the thicknesses, where distances are discretized to an integer number of milli-Angstroms
            tgcd=reduce(gcd,[rint(l.thickness/1e-3) for l in stack])*1e-3

            # The spacing to use must be an integer divisor of that gcd,
            # and ceil guarantees it will be smaller than max_dz
            dz=tgcd/ceil(tgcd/max_dz)

            # Figure out how many points this is, then use linspace to create the mesh
            totthick=sum([l.thickness for l in stack])
            N=rint(totthick/dz)+1
            fixed_positions=np.linspace(0,totthick,num=N,endpoint=True)

            # List all the interface indices
            interface_indices=np.rint(np.cumsum([l.thickness for l in stack])/dz)

        else:
            # Implement the max_dz requirement by adding it to the refinements list
            if refinements:
                refinements = np.vstack([np.array(refinements), [0, max_dz, 1]])
            else:
                refinements = np.array([[0, max_dz, 1]])

            # List of z points which have been finalized (ie are behind the most recent interface)
            fixed_positions = [0]

            # List of indices for interfaces
            interface_indices = []

            # z point of left interface of currently building region
            zl = 0

            # For each region
            for layer in stack:

                # z point of right interface of currently building region
                zr = zl + layer.thickness

                # start from the left interface
                z = zl

                # List of z points are being built (ie after the most recently passed interface)
                variable_positions = []

                # Build until we pass right interface
                while True:
                    # The maximal allowed dz is the minimum of the the refinement criteria
                    dz = np.min(refinements[:, 1] * refinements[:, 2] ** np.abs(z - refinements[:, 0]))

                    # Extend the mesh
                    z += dz
                    variable_positions += [z]
                    if z > zr - 1e-10: break

                # Once we reach the right interface, shrink the mesh uniformly for a perfect fit
                pos = (np.array(variable_positions) - zl) * layer.thickness / (z - zl) + zl

                # Append the new z points to the list of fixed locations
                fixed_positions += list(pos)

                # Record the index of the interface
                interface_indices += [len(fixed_positions) - 1]

                # The left index of the next interface is the right index of the current
                zl = zr

        # Convert the built z list to numpy array
        self._zp = np.array(fixed_positions)
        self._dzp = np.diff(self._zp)

        # Compile a list of interfaces for z
        interface_indices=np.array(interface_indices,dtype=int)
        # Each element is a tuple of the form (index, left layer, right layer)
        self._interfacesp = list(zip(interface_indices[:-1], stack[:-1], stack[1:]))
        # Compile a list of interfaces for zp
        # Each element is a tuple of the form (lindex,rindex, left layer, right layer)
        self._interfacesm= list(zip(interface_indices[:-1] - 1, interface_indices[:-1], stack[:-1], stack[1:]))

        # Keep the layer info
        self._layers = stack

        # Also keep the z's in-between mesh points
        self._zm = (self._zp[:-1] + self._zp[1:]) / 2
        self._dzm = np.array([0.] * len(self._zp))
        self._dzm[1:-1] = np.diff(self._zm)
        self._dzm[[0, -1]] = self._dzm[[1, -2]]

        # interpolate the z -> index mapping
        self._zp2i_interp = interp1d(self._zp, np.arange(len(self._zp)))
        # interpolate the zp -> index mapping
        self._zm2i_interp = interp1d(self._zm, np.arange(len(self._zm)))

        # This is the whole world
        self._supermesh = None
        self._submeshes = []

        # Store functions which live on this mesh
        self._functions = {}

        self.pmdb=stack._pmdb

    def indexp(self, zp):
        r""" Finds the index of the point mesh location nearest to ``zp``.

        :param zp: :math:`z` position
        :return: an index into the point mesh
        """
        return np.rint(self._zp2i_interp(zp)).astype(int)
    def indexm(self, zm):
        r""" Finds the index of the mid mesh location nearest to ``zm``.

        :param zm: :math:`z` position
        :return: an index into the mid mesh
        """
        return np.rint(self._zm2i_interp(zm)).astype(int)

    def plot_mesh(self):
        """ Plots a 1-D representation of the mesh for visual inspection.
        """

        # Make a long, thin figure
        mpl.figure(figsize=(8, 2))

        # Collect the z values at interfaces
        ipoints = self._zp[[0] + [i[0] for i in self._interfaces] + [len(self._zp) - 1]]

        # Draw a vertical line and label for each interface
        for ii, i in enumerate(ipoints):
            if ii in [0, len(ipoints) - 1]:
                mpl.vlines(i, -.5, .25, linestyles='dashed')
            else:
                mpl.vlines(i, -.5, .25)
            mpl.text(i, .25, "{:.3g}".format(i),
                     horizontalalignment='center', verticalalignment='bottom')

        # Draw a material label over each region
        for i, m in zip((ipoints[1:] + ipoints[:-1]) / 2, [l.name for l in self._layers]):
            mpl.text(i, .1, m, horizontalalignment='center')

        # Draw a small vertical line for each mesh point
        mpl.vlines(self._zp, -.05, .05)

        # Fit the xlimits to the mesh
        mpl.xlim(self._zp[0], self._zp[-1] + .1)

        # Fit the ylimits to the drawn lines
        mpl.ylim(-.05, .5)

        # Get rid of extra axes and ticks
        mpl.gca().get_yaxis().set_visible(False)
        mpl.gca().spines['top'].set_visible(False)
        mpl.gca().spines['right'].set_visible(False)
        mpl.gca().spines['left'].set_visible(False)
        mpl.gca().get_xaxis().tick_bottom()
        mpl.gca().get_yaxis().tick_left()

        mpl.title('Total mesh points: {:d}'.format(len(self._zp)))
        mpl.tight_layout()

    def __contains__(self,key):
        r""" True iff there is a function ``key`` defined on this mesh."""
        return key in self._functions

    def __iter__(self):
        r""" Iterate through functions defined on this mesh."""
        return self._functions.__iter__()

    def __getitem__(self, key):
        r""" Get by name a function defined on this mesh."""
        return self._functions[key]

    def __setitem__(self, key, value):
        r""" Update (or create) a function on this mesh.  Propagates to any submeshes."""
        if key in self._functions:
            self._functions[key][:] = value
        else:
            assert isinstance(value,Function), "Must be a mesh.functions.Function"
            self._functions[key] = value
            for sm in self._submeshes:
                sm._functions[key]=value.restrict(sm)

    @property
    def zp(self):
        r""" the locations of the point mesh as a numpy array"""
        return self._zp

    @property
    def zm(self):
        r""" the locations of the mid mesh as a numpy array"""
        return self._zm

    @property
    def dzp(self):
        r""" the spacing between locations in the point mes as a numpy arrayh"""
        return self._dzp

    @property
    def dzm(self):
        r""" the spacing between locations in the mid mes as a numpy arrayh"""
        return self._dzm

    @property
    def interfaces_point(self):
        r""" list of interfaces and adjacent :py:class:`~pynitride.poissolve.mesh.Layer`'s on the point mesh.

        :return: each element is a tuple of the form ``(index, layer_to_left, layer_to_right)``
        """
        return self._interfacesp
    @property
    def interfaces_mid(self):
        r""" list of interfaces and adjacent :py:class:`~pynitride.poissolve.mesh.Layer`'s on the mid mesh.

        :return: each element is a tuple of the form
            ``(index_to_left, index_to_right, layer_to_left, layer_to_right)``
        """
        return self._interfacesm

    def submesh(self, zbounds):
        r""" Returns a :py:class:`~pynitride.poissolve.mesh.SubMesh` viewing a range of this mesh.

        The range is specified by desired :math:`z` locations.  If you want to specify exact indices instead, use the
        :py:class:`~pynitride.poissolve.mesh.SubMesh` constructor directly.

        :param zbounds: two-element tuple of :math:`z` locations to start and stop the submesh (inclusive)
        :return: a :py:class:`~pynitride.poissolve.mesh.SubMesh` which views this mesh in the desired range
        """
        return SubMesh(self, self.indexp(zbounds[0]), self.indexp(zbounds[1]) + 1)

    # Finish making and testing save and load
    #def save(self,filename):
    #    r""" Save the mesh, its submeshes, and all functions defined on it to a file.
    #    :param filename: filename to save the mesh, must end with ".msh"
    #    """
    #    assert filename[-4:]==".msh", "Only .msh format currently supported."
    #    with open(filename,"wb") as f:
    #        pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)

    #@staticmethod
    #def load(filename):
    #    assert filename[-4:]==".msh", "Only .msh format currently supported."
    #    with open(filename,"rb") as f:
    #        return pickle.load(f)

class SubMesh(Mesh):
    """ Represents a Mesh restricted to to particular segment.

    A submesh is essentially just a view of a mesh in that segment (data is shared).
    """

    """ Construct a submesh.

    Arguments:
        mesh: the Mesh from which to draw data
        start: the start index (inclusive) of the slice
        stop: the stop index (exclusive) of the slice

    """

    def __init__(self, mesh, start, stop):

        if start is None: start=0
        if stop is None: stop=len(mesh.zp)

        self._slicep = slice(start, stop)
        self._slicem = slice(start, stop - 1)

        self._zp = mesh._zp[self._slicep]
        self._zm = mesh._zm[self._slicem]
        self._dzp = mesh._dzp[self._slicem]
        self._dzm = mesh._dzm[self._slicep]

        self._interfacesp = [(i - start, ll, lr) for i, ll, lr in mesh.interfaces_point if (i > start and i < stop - 1)]
        # THIS IS A HORRIBLE HACK.  I'M SORRY, FUTURE SAM.
        if len(self.interfaces_point):
            self._layers = EpiStack(*[ll for i, ll, lr in self.interfaces_point] + [self.interfaces_point[-1][2]])
        else:
            self._layers=EpiStack(next(ll for i,ll,lr in (mesh.interfaces_point+[[start+1,mesh._layers[-1],None]]) if i > start))

        self._functions = { k: f.restrict(self) for k, f in mesh._functions.items()}

        self._supermesh = mesh
        self._submeshes = []
        if self not in mesh._submeshes:
            mesh._submeshes += [self]

        # interpolate the zp -> index mapping
        self._zp2i_interp = interp1d(self._zp, np.arange(len(self._zp)))
        # interpolate the zm -> index mapping
        self._zm2i_interp = interp1d(self._zm, np.arange(len(self._zm)))

        self.pmdb=mesh.pmdb

class Function(np.ndarray):
    def __new__(cls, *args, **kwargs):
        raise NotImplementedError

    def __array_finalize__(self, obj):
        if obj is None: return

        self.mesh = getattr(obj, 'mesh', "View casting from ndarray not supported.")


class PointFunction(Function):
    def __new__(cls, mesh, value=np.NaN, dtype='float', empty=False):

        # If the user just wants an empty array, the shape of an element is specified by empty
        if empty:
            vshape=list(empty)
            obj = np.empty(vshape + list(mesh.zp.shape), dtype=dtype).view(cls)
            obj.mesh=mesh
            return obj

        # Otherwise, read thhe shape from the given value
        value = np.asarray(value,dtype=dtype)
        vshape=list(value.shape)

        # If the shape matches up to the mesh already, go ahead and just view that value as the PointFunction
        if len(value.shape) and value.shape[-1]==mesh.zp.shape[0]:
            obj=value.view(cls)
            obj.mesh = mesh
            return obj

        # If the shape is one dimensional (but doesn't match the mesh, see above), reshape it to 2d
        # with the length=1 along the last axis.
        elif len(value.shape)==1:
            value=np.reshape(value,(len(value),1))

        # Try to form a full array by repeating this element len(mesh.z) times.
        try:
            obj = np.full(vshape + list(mesh.zp.shape), value, dtype=dtype).view(cls)
            obj.mesh = mesh
            return obj
        except:
            raise Exception("Given arr of shape {} is not compatible with given point mesh of size {}".format(value.shape, mesh.zp.shape[0]))

    def plot(self,*args,**kwargs):
        mpl.plot(self.mesh.zp, self, *args, **kwargs)

    # def __array_prepare__(self, out_arr, context=None):
    #    assert out_arr.shape==self.mesh.z.shape,\
    #        "Can't combine Functions of different mesh sizes"
    #    out_arr.shape=self.mesh.z.shape
    #    return out_arr

    def differentiate(self):
        return MidFunction(self.mesh, np.diff(self, axis=-1) / self.mesh.dzp)

    # provide a non-cumsum, just sum, version for efficiency when that's all that's wanted
    def integrate(self, flipped=False):
        return np.cumsum(
            (self * self.mesh.dzm).T[:-1].T
            if not flipped
            else np.flipud(-self * self.mesh.dzm).T[:-1].T,
            axis=-1).view(MidFunction)

    def restrict(self, submesh):
        # doesn't check that submesh and mesh are compatible
        return type(self)(submesh, self.T[submesh._slicep].T)


class MidFunction(Function):
    def __new__(cls, mesh, value=np.NaN, dtype='float'):
        value = np.asarray(value,dtype=dtype)
        vshape=list(value.shape)
        if len(value.shape) and value.shape[-1]==mesh.zm.shape[0]:
            obj=value.view(cls)
            obj.mesh = mesh
            return obj
        elif len(value.shape)==1:
            value=np.array([value]).T
        try:
            obj = np.full(vshape+list(mesh.zm.shape), value, dtype=dtype).view(cls)
        except:
            # THIS MESSAGE MIGHT BE CONFUSING BECAUSE THE MESH IS ACTUALLY CALLED SIZE mesh.z
            raise Exception("Given arr of shape {} is not compatible with given mid mesh of size {}".format(value.shape,mesh.zm.shape[0]))
        obj.mesh = mesh
        return obj

    def plot(self,*args,**kwargs):
        mpl.plot(self.mesh.zm,self,*args,**kwargs)
        # def __array_prepare__(self, out_arr, context=None):
    #    assert out_arr.shape==self.mesh.zp.shape,\
    #        "Can't combine Functions of different mesh sizes"
    #    out_arr.shape=self.mesh.zp.shape
    #    return out_arr

    def differentiate(self, fill_value=np.NaN):
        pf = PointFunction(self.mesh,empty=np.array(self.T[0].shape))
        pf.T[1:-1] = (np.diff(self,axis=-1) / self.mesh.dzm[1:-1]).T
        pf.T[[0, -1]] = fill_value
        return pf

    def integrate(self, flipped=False):
        # if output is None:
        output = PointFunction(self.mesh, value=0.0)
        np.cumsum(
            self * self.mesh._dz if not flipped else np.flipud(-self * self.mesh._dz),
            out=(output[1:] if not flipped else np.flipud(output[:-1])), axis=-1)
        return output

    def to_point_function(self, interp='z'):
        if interp == 'unweighted':
            newshape=list(self.shape)
            newshape[-1]+=1
            arr = np.empty(newshape).T
            arr[1:-1] = (self.T[1:] + self.T[:-1]) / 2
            arr[[0, -1]] = arr[[1, -2]]
            arr=arr.T
        if interp == 'z':
            arr = interp1d(self.mesh.zm, self,
                           fill_value='extrapolate')(self.mesh.zp)
        return PointFunction(self.mesh, arr)

    def restrict(self, submesh):
        # doesn't check that submesh and mesh are compatible
        return type(self)(submesh, self.T[submesh._slicem].T)


def ConstantFunction(mesh, val, dtype='float', pos='point'):
    from numpy.lib.stride_tricks import as_strided
    x = np.array(val, order='C', dtype=dtype)
    newshape = list(x.shape) + [mesh.zp.shape[0] if pos == 'point' else mesh.zm.shape[0]]
    newstrides = list(x.strides) + [0]
    arr=as_strided(np.array(x), shape=newshape, strides=newstrides)
    return {'point': PointFunction, 'mid': MidFunction}[pos](mesh,value=arr)


def MaterialFunction(mesh, prop, default=None,pos='mid'):
    # could make this more efficient by directly interpolating if Point case?
    # this function almost duplicates RegionFunction...

    ptcounts = np.diff([0] + [i for i, ll, lr in mesh.interfaces_point] + [len(mesh.zp) - 1])
    arr = []

    propfunc = (lambda i: prop(mesh._layers[i].material)) \
        if callable(prop) \
        else (lambda i: mesh._layers[i].get(prop,default=default))

    for i, ptc in enumerate(ptcounts):
        arr += [propfunc(i)] * ptc

    out = MidFunction(mesh, np.array(arr).T)
    if pos == "point":
        return out.to_point_function()
    else:
        return out


def RegionFunction(mesh, prop, pos='mid'):
    # could make this more efficient by directly interpolating if Point case?

    ptcounts = np.diff([0] + [i for i, ll, lr in mesh.interfaces_point] + [len(mesh.zp) - 1])
    arr = []

    propfunc = (lambda i: prop(mesh._layers[i].name)) \
        if callable(prop) \
        else (lambda i: mesh._layers[i][prop])
    for i, ptc in enumerate(ptcounts):
        arr += [propfunc(i)] * ptc

    out = MidFunction(mesh, arr)
    if pos == "point":
        return out.to_point_function()
    else:
        return out


def DeltaFunction(mesh, z, height=1, i=None, pos='point'):
    func={'point': PointFunction, 'mid': MidFunction}[pos](mesh,0.0)
    i={'point': mesh.indexp, 'mid': mesh.indexp}[pos](z) if i is None else i
    func[i]=height/{'point':mesh._dzp[i], 'mid':mesh.dzm[i]}[pos]
    return func