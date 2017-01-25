# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:15:41 2017

@author: sam
"""

import matplotlib.pyplot as mpl
import numpy as np
from scipy.interpolate import interp1d
import pickle

from pynitride.poissolve.materials import Material
from pynitride.poissolve.mesh.functions import Function


class Layer():
    def __init__(self, name, matname, thickness):
        self._name = name
        self._matname = matname
        self._mat = Material(matname)
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
        return self._mat.get(key,default)

    def __getitem__(self, key):
        return self._mat[key]


class EpiStack():
    def __init__(self,*args,surface=None):
        if isinstance(args[0],Layer):
            self._layers=args
        else:
            self._layers=[Layer(l[0], l[1], l[2]) if len(l) == 3 else Layer(l[0], l[0], l[1]) for l in args]
        self._surface=surface

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
    """ Generates and manages a 1-D mesh."""

    def __init__(self, stack, max_dz, refinements=[], uniform=False):
        """ Constructs a non-uniform mesh with vertices aligned to the interfaces of a material stack.
        
        The algorithm is to step through the regions one-by-one and build the mesh point-by-point.  At a
        point z, the mesh will be extended by adding a maximal dz which satisfies all the refinement criteria
        (as a simplification, the criteria are evaluated at z, not z+dz).  When the mesh being constructed
        passes an interface, the entire mesh built since the last interface is shrunk uniformly so that the
        last mesh point matches that interface.
        
        Refinement criteria are given as a max dz0 near a refinement point z0, and an exponential growth rate
        for that dz.  This makes it easy to evaluate at any z what is the maximal allowed dz.  As mentioned,
        when the mesh is built, the criteria are evaluated at the most recent point z, not at z+dz.  So it is
        possible that dz may be slightly larger than the refinement criteria if the limiting refinement is to
        the right.  If the mesh is sufficiently dense, the refinement criteria will be approximately satisfied.
        But, because of this detail, note that these criteria are targets (which will be nearly hit), not
        guarantees.  

        Arguments:
            stack: the EpiStack of the device
            
            max_dz: the maximum mesh spacing allowed globally.
            
            refinements: a list of spots where the mesh should be refined.  Each element is a triple
                (z0,dz0,r), where z0 is the location that should be refined, dz0 is the target mesh spacing
                in the region of z0, and r is the target rate at which the mesh spacing is allowed to
                exponentially increase moving away from z0.  ie each refinement enforces a constraint of
                the form dz < dz0 * r^|z-z0|.

        """
        if uniform:
            from math import gcd,ceil
            from functools import reduce
            rint=lambda x: int(round(x))
            tgcd=reduce(gcd,[rint(l.thickness/1e-10) for l in stack])*1e-10
            totthick=sum([l.thickness for l in stack])
            dz=tgcd/ceil(tgcd/max_dz)
            N=rint(totthick/dz)+1
            fixed_positions=np.linspace(0,totthick,num=N,endpoint=True)
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
        self._z = np.array(fixed_positions)
        self._dz = np.diff(self._z)

        # Compile a list of interfaces for z
        interface_indices=np.array(interface_indices,dtype=int)
        # Each element is a tuple of the form (index, left layer, right layer)
        self._interfaces = list(zip(interface_indices[:-1], stack[:-1], stack[1:]))
        # Compile a list of interfaces for zp
        # Each element is a tuple of the form (lindex,rindex, left layer, right layer)
        self._interfacesp= list(zip(interface_indices[:-1]-1, interface_indices[:-1], stack[:-1], stack[1:]))

        # Keep the layer info
        self._layers = stack

        # Also keep the z's in-between mesh points
        self._zp = (self._z[:-1] + self._z[1:]) / 2
        self._dzp = np.array([0.] * len(self._z))
        self._dzp[1:-1] = np.diff(self._zp)
        self._dzp[[0, -1]] = self._dzp[[1, -2]]

        # interpolate the z -> index mapping
        self._z2i_interp = interp1d(self._z, np.arange(len(self._z)))
        # interpolate the zp -> index mapping
        self._zp2i_interp = interp1d(self._zp, np.arange(len(self._zp)))

        # This is the whole world
        self._supermesh = None
        self._submeshes = []

        # Store functions which live on this mesh
        self._functions = {}

    def index(self, z):
        return np.rint(self._z2i_interp(z)).astype(int)
    def indexp(self, zp):
        return np.rint(self._zp2i_interp(zp)).astype(int)

    def plot_mesh(self):
        """ Plots a 1-D representation of the mesh for visual inspection.
        """

        # Make a long, thin figure
        mpl.figure(figsize=(8, 2))

        # Collect the z values at interfaces
        ipoints = self._z[[0] + [i[0] for i in self._interfaces] + [len(self._z) - 1]]

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
        mpl.vlines(self._z, -.05, .05)

        # Fit the xlimits to the mesh
        mpl.xlim(self._z[0], self._z[-1] + .1)

        # Fit the ylimits to the drawn lines
        mpl.ylim(-.05, .5)

        # Get rid of extra axes and ticks
        mpl.gca().get_yaxis().set_visible(False)
        mpl.gca().spines['top'].set_visible(False)
        mpl.gca().spines['right'].set_visible(False)
        mpl.gca().spines['left'].set_visible(False)
        mpl.gca().get_xaxis().tick_bottom()
        mpl.gca().get_yaxis().tick_left()

        mpl.title('Total mesh points: {:d}'.format(len(self._z)))
        mpl.tight_layout()

    def __contains__(self,key):
        return key in self._functions

    def __iter__(self):
        return self._functions.__iter__()

    def __getitem__(self, key):
        return self._functions[key]

    def __setitem__(self, key, value):
        if key in self._functions:
            self._functions[key][:] = value
        else:
            assert isinstance(value,Function), "Must be a poissolve.mesh_function.Function"
            self._functions[key] = value
            for sm in self._submeshes:
                sm[key]=value.restrict(sm)

    def plot_function(self, key, *args, **kwargs):
        self._functions[key].plot(*args, **kwargs)

    @property
    def z(self):
        return self._z

    @property
    def zp(self):
        return self._zp

    @property
    def interfaces_point(self):
        return self._interfaces
    @property
    def interfaces_mid(self):
        return self._interfacesp

    def submesh(self, zbounds):
        #return SubMesh(self, *list(self.index(zbounds)))
        return SubMesh(self, self.index(zbounds[0]),self.index(zbounds[1])+1)

    def save(self,filename):
        assert filename[-4:]==".msh", "Only .msh format currently supported."
        with open(filename,"wb") as f:
            pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        assert filename[-4:]==".msh", "Only .msh format currently supported."
        with open(filename,"rb") as f:
            return pickle.load(f)


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
        if stop is None: stop=len(mesh.z)

        self._slice = slice(start, stop)
        self._slicep = slice(start, stop - 1)

        self._z = mesh._z[self._slice]
        self._zp = mesh._zp[self._slicep]
        self._dz = mesh._dz[self._slicep]
        self._dzp = mesh._dzp[self._slice]

        self._interfaces = [(i - start, ll, lr) for i, ll, lr in mesh._interfaces if (i > start and i < stop - 1)]
        # THIS IS A HORRIBLE HACK.  I'M SORRY, FUTURE SAM.
        if len(self._interfaces):
            self._layers = EpiStack(*[ll for i, ll, lr in self._interfaces] + [self._interfaces[-1][2]])
        else:
            self._layers=EpiStack(next(ll for i,ll,lr in (mesh._interfaces+[[start+1,mesh._layers[-1],None]]) if i > start))

        self._functions = {
            k: f.restrict(self) for k, f in mesh._functions.items()
            }

        self._supermesh = mesh
        self._submeshes = []
        if self not in mesh._submeshes:
            mesh._submeshes += [self]

        # interpolate the z -> index mapping
        self._z2i_interp = interp1d(self._z, np.arange(len(self._z)))
        # interpolate the zp -> index mapping
        self._zp2i_interp = interp1d(self._zp, np.arange(len(self._zp)))


if __name__ == '__main__':
    from runpy import run_path

    run_path("./tests/test_structure.py", run_name='__main__')

    print("HELLO")
    epistack = EpiStack(['GaN', 2.5], ['AlN', 5], ['GaN', 17], ['AlN', 30])
    m = Mesh(epistack, max_dz=1, refinements=[[7.5, .1, 1.1], [24, .25, 1.2]])
    sm = m.submesh([5, 27])

    mpl.close('all')
    m.plot_mesh()
    mpl.gcf().canvas.set_window_title('Global Mesh')
    sm.plot_mesh()
    mpl.gcf().canvas.set_window_title('Schrodinger Mesh')
    mpl.show()
    print("THERE")
