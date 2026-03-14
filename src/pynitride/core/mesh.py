# -*- coding: utf-8 -*-
r""" Meshing, submeshing, and manipulating functions defined on meshes."""

import matplotlib.pyplot as mpl
import numpy as np
from matplotlib import pyplot as mpl
from scipy.interpolate import interp1d
from math import gcd,ceil
from functools import reduce
from pynitride import log
from scipy.special import lambertw as W
from pynitride.core.fem import assemble_load_matrix
from copy import deepcopy
import warnings

class MaterialBlock():
    def __init__(self,name,matsys,layers):
        """ Represents a region of the simulation consisting of one material system

        Args:
            name: arbitrary name for this region
            matsys: the :class:`~pynitride.physics.material.MaterialSystem`
            layers: a list of layers inside this block
                As a convenience for supplying stacks, layers of thickness=0 can be supplied without causing issues;
                they will just be silently filtered out on initialization.
        """
        self.name=name
        self.matsys=matsys
        layers=[l for l in layers if l.thickness>0]
        self.layers=layers
        for l in layers:
            l._matblock=self

    def place(self,mesh,interface_indices):
        """ Places the block onto the submesh, and places all its layers

        Args:
            mesh: the submesh assigned to this block
            interface_indices: list of interfaces (including start and end)
                along which the layers will be placed

        """
        self._mesh=mesh
        mesh.name=self.name
        for k,v in self.matsys._defaults.items():
            mesh[k]=MidFunction(mesh,v)
        for lay,l,r in zip(self.layers,interface_indices[:-1],interface_indices[1:]):
            lay.place(SubMesh(mesh,'',l,r+1))

    @property
    def mesh(self):
        """ The submesh owned by this material block."""
        return self._mesh

    def get(self, item, destmesh=None, destfunc=None):
        """ Returns the requested function retrieving from the material if necessary.

        Typical use with no `destmesh` or `destfunc`:
        If the function is already defined on this mesh, returns it.
        Otherwise, if it's available from the material system, regturn it

        Other use (internal by :class:`Mesh`):
        If a `destfunc` is supplied, the results will be filled into the proper slice of `destfunc`,
        and `destfunc` will be returned.

        If a `destfunc` is not supplied, but `destmesh` is (and that `destmesh` is not the mesh of this block),
        a function will be created matching the `dtype` found for this key on `destmesh`, and the values will be filled
        into the proper slice, and the function will be returned

        Args:
            item: the key (string) sought
            destmesh: (see above)
            destfunc: (see above)

        Returns:
            a :class:`Function`

        """
        if destfunc is not None:
            destmesh=destfunc.mesh
        elif destmesh is None:
            destmesh=self._mesh

        # Get the subfunc from matsys
        if item in self._mesh._functions:
            subfunc=self._mesh._functions[item]
            if destmesh==self._mesh:
                return subfunc
        else:
            subfunc=self.matsys.get(self._mesh,item)
            if destmesh==self._mesh:
                return subfunc

        # Get the func if it's defined on this mesh
        if item in destmesh._functions:
            func=destmesh[item]

        # Or if we haven't made the global func yet, make it from this one
        elif destfunc is None:
            func=MidFunction(destmesh, dtype=subfunc.dtype, empty=subfunc.shape[:-1])
        else:
            func=destfunc

        # Figure out the ranges where the desination mesh and material block overlap
        globalstart=max(self._mesh._global_slicem.start,destmesh._global_slicem.start)
        globalstop =min(self._mesh._global_slicem.stop ,destmesh._global_slicem.stop )

        # Fill in the relevant part of the function
        func[globalstart-destmesh._global_slicem.start:globalstop-destmesh._global_slicem.start]=\
            subfunc.T[globalstart-self._mesh._global_slicem.start:globalstop-self._mesh._global_slicem.start].T
        return func

    def __contains__(self, item):
        return (item in self.matsys) or (item in self._mesh._functions)
    
    def update(self,destmesh,reason):
        self.matsys.update(destmesh,reason)

class Layer():
    def __init__(self, name, thickness):
        """ A chunk of simulation domain which will be guaranteed to end on node-points

        Args:
            name: an arbitrary name for the layer
            thickness: the thickness of the layer
        """
        self.name = name
        self.thickness = thickness

    def place(self,mesh):
        """ Places the layer onto the mesh (called by :class:`MaterialBlock.place`"""
        self._mesh=mesh
        mesh.name=self.name

    @property
    def mesh(self):
        """ The submesh owned by this layer"""
        return self._mesh

    @property
    def matblock(self):
        """ The material block which contains this layer"""
        return self._matblock

    def __getitem__(self, key):
        return self._mesh[key]

class UniformLayer(Layer):
    def __init__(self, name, thickness, **kwargs):
        """ A uniform chunk of simulation domain which will be guaranteed to end on node-points

        That is to say, the properties specified by kwargs will be uniform

        Args:
            name: an arbitrary name for the layer
            thickness: the thickness of the layer
            kwargs: any other properties specified uniformly in the region
        """
        super().__init__(name,thickness)
        self._setproperties=kwargs

    def place(self,mesh):
        """ Places the layer onto the mesh (called by :class:`MaterialBlock.place`, filling in uniform values."""
        super().place(mesh)
        for k,v in self._setproperties.items():
            if type(v) is bool:
                dtype='bool'
            else:
                dtype='float'
            mesh[k]=MidFunction(mesh,value=v,dtype=dtype)

class GradedLayer(Layer):
    def __init__(self, name, thickness, **kwargs):
        """ A uniform chunk of simulation domain which will be guaranteed to end on node-points

        That is to say, the properties specified by kwargs will be uniform

        Args:
            name: an arbitrary name for the layer
            thickness: the thickness of the layer
            kwargs: any other properties specified uniformly in the region
        """
        super().__init__(name,thickness)
        self._setproperties=kwargs

    def place(self,mesh):
        """ Places the layer onto the mesh (called by :class:`MaterialBlock.place`, filling in uniform values."""
        super().place(mesh)
        for k,v in self._setproperties.items():
            if k[:6]=="start_":
                k=k[6:]
                vstart=v
                vstop=self._setproperties["stop_"+k]
                mesh[k]=LinearFunction(mesh,vstart,vstop,pos='node').tmf()
            elif "stop_" in k:
                continue
            else:
                if type(v) is bool:
                    dtype='bool'
                else:
                    dtype='float'
                mesh[k]=MidFunction(mesh,value=v,dtype=dtype)


class Mesh():

    def __init__(self, stack, max_dz, refinements=[], uniform=False, boundary=["GenericMetal","thick"]):
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

        Args:
            stack: the :py:class:`~pynitride.poissolve.mesh.EpiStack` representing the device
            max_dz: (number) the maximum mesh spacing allowed globally.
            refinements: a list of spots where the mesh should be refined.
                Each element is a triple :math:`(z^p_0,dz^p_0,r)`,
                where :math:`z_0` is the location that should be refined,
                :math:`dz^p_0` is the target mesh spacing in the region of :math:`z_0`,
                and :math:`r` is the target rate at which the mesh spacing is allowed to exponentially increase
                moving away from :math:`z_0`. ie each refinement enforces a constraint of the form
                :math:`dz \lesssim dz^p_0  r^{|z-z_0|}`.
                Note that, when creating a uniform mesh, the only effect of this argument is to reduce ``max_dz`` if a
                refinement with `dz^p_0` tighter than ``max_dz`` is included.
            uniform: if True, keep the spacing uniform instead of applying the complicated meshing above
            boundary: a two-tuple of boundary conditions.  At present, the second element must be "thick", but the first
                element can be the name of a metal which which the top material knows its Schottky barrier, or can be a
                number directly specifying the barrier

        """
        self._boundary=boundary
        self.ztrans=-1
        self._matblocks=stack
        self._layers = layers = sum([mb.layers for mb in stack],[])

        # Parse refinements
        for r in refinements:
            zr=r[0]
            if isinstance(zr,str):
                l1n,l2n=zr.split("/")
                try:
                    #print("Layers are \""+"\", \"".join([l.name for l in layers])+"\"")
                    l1,_=next((i,l) for i,l in enumerate(layers) if l.name==l1n)
                    l2,_=next((i,l) for i,l in enumerate(layers) if l.name==l2n)
                except:
                    raise Exception("A layer ({} or {}) was not found for refinement.".format(l1n,l2n))
                if (l2-l1)>1: raise Exception("Interface {} not found".format(zr))
                r[0]=np.cumsum([l.thickness for l in layers])[min(l1,l2)]


        # Make a uniform mesh
        if uniform or (refinements==[]):

            # Get the maximum spacing from the tightest of max_dz and the refinements
            max_dz=min([max_dz]+[r[1] for r in refinements])

            # Quick util to round and convert to an integer type
            rint=lambda x: int(round(x))

            # Get the gcd of the thicknesses, where distances are discretized to an integer number of milli-Angstroms
            tgcd=reduce(gcd,[rint(l.thickness/1e-3) for l in layers])*1e-3

            # The spacing to use must be an integer divisor of that gcd,
            # and ceil guarantees it will be smaller than max_dz
            dz=tgcd/ceil(tgcd/max_dz)

            # Figure out how many points this is, then use linspace to create the mesh
            totthick=self.thickness=sum([l.thickness for l in layers])
            N=rint(totthick/dz)+1
            fixed_positions=np.linspace(0,totthick,num=N,endpoint=True)

            # List all the interface indices
            interface_indices=np.rint(np.cumsum([l.thickness for l in layers])/dz)

        else:

            # List of z points which have been finalized (ie are behind the most recent interface)
            fixed_positions = [0]

            # List of indices for interfaces
            interface_indices = []

            # z point of left interface of currently building region
            zl = 0

            # For each region
            for layer in layers:

                # z point of right interface of currently building region
                zr = zl + layer.thickness

                # start from the left interface
                z = zl

                # List of z points are being built (ie after the most recently passed interface)
                variable_positions = []

                # Build until we pass right interface
                while True:

                    # The maximal allowed dz is the minimum of the the refinement criteria
                    dz=max_dz
                    with np.errstate(over='ignore'):
                        for zref,zminref,rref in refinements:
                            if zref-z<=0:
                                dz=min(dz,zminref*rref**abs(zref-z))
                            else:
                                # the second entry in the below list is the
                                # max dz such that dz < zminref x r ^ |z+dz|, assuming z+dz<zref
                                # (cool tricks with the lambert W function)
                                dz=min([dz,abs(1/np.log(rref)*W(np.log(rref)/1*zminref*rref**(zref-z))),zref-z])

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

        interface_indices=interface_indices[:-1]

        # Convert the built z list to numpy array
        self._zn = np.array(fixed_positions)
        self._dzn = np.diff(self._zn)

        # Compile a list of interfaces for z
        interface_indices=np.array(interface_indices,dtype=int)
        # Each element is a tuple of the form (index, left layer, right layer)
        self._interfacesn = list(zip(interface_indices, layers[:-1], layers[1:]))
        # Compile a list of interfaces for zn
        # Each element is a tuple of the form (lindex,rindex, left layer, right layer)
        self._interfacesm= list(zip(interface_indices - 1, interface_indices, layers[:-1], layers[1:]))

        # Also keep the z's in-between mesh points
        self._zm = (self._zn[:-1] + self._zn[1:]) / 2
        self._dzm = np.array([self._dzn[0]] * len(self._zn))
        self._dzm[1:-1] = np.diff(self._zm)
        self._dzm[[0, -1]] = self._dzn[[0, -1]]/2

        if len(self._zm)>1:
            # interpolate the zn -> index mapping
            self._zn2i_interp = interp1d(self._zn, np.arange(len(self._zn)))
            # interpolate the zm -> index mapping
            self._zm2i_interp = interp1d(self._zm, np.arange(len(self._zm)))

        # Store functions which live on this mesh
        self._functions = {}
        self._requested_functions = {}
        self._submeshes= []

        self._global_slicen=slice(0,len(self._zn))
        self._global_slicem=slice(0,len(self._zm))


        # This is the whole world
        self.name='global'
        self._supermesh = None
        leftindices=[0]+interface_indices.tolist()
        rightindices=interface_indices.tolist()+[len(self._zn)-1]
        ill=-1
        ilr=-1
        for i,mb in enumerate(self._matblocks):
            ill=ilr+1
            ilr=ill+len(mb.layers)-1
            ml=leftindices[ill]
            mr=rightindices[ilr]
            mbii=leftindices[ill:(ill+len(mb.layers))]+[mr]
            sm=SubMesh(self, '', ml, mr+1)
            mb.place(sm,interface_indices=list(np.array(mbii)-ml))

        self.Nn=len(self._zn)
        """ Number of node points"""
        self.Nm=len(self._zm)
        """ Number of mid points"""

        self.zeros_nod=NodFunction(self,0)
        """ A 1-D all-zeros node function on this mesh"""
        self.zeros_mid=MidFunction(self,0)
        """ A 1-D all-zeros mid function on this mesh"""
        self.ones_nod=NodFunction(self,1)
        """ A 1-D all-ones node function on this mesh"""
        self.ones_mid=MidFunction(self,1)
        """ A 1-D all-ones mid function on this mesh"""

        self._metric=assemble_load_matrix(self.ones_mid,self.dzn,n=1,dirichelet1=False,dirichelet2=False)

    def __repr__(self):
        return "<Mesh("+str(self.Nn) + ") \"" + str(self.name) + "\">"

    def indexn(self, zn):
        r""" Finds the index of the node mesh location nearest to ``zn``.

        Args:
            zn: :math:`z` position
        Returns:
            an index into the node mesh
        """
        return np.rint(self._zn2i_interp(zn)).astype(int)
    def indexm(self, zm):
        r""" Finds the index of the mid mesh location nearest to ``zm``.

        Args:
            zm: :math:`z` position
        Returns:
            an index into the mid mesh
        """
        return np.rint(self._zm2i_interp(zm)).astype(int)

    def matblock(self,name):
        """ Returns the material block with the given name (error if not found)"""
        return next(mb for mb in self._matblocks if mb.name==name)

    def plot_mesh(self,xlim=None):
        """ Plots a 1-D representation of the mesh for visual inspection."""

        # Make a long, thin figure
        mpl.figure(figsize=(8, 2))

        # Collect the z values at interfaces
        ipoints = self._zn[[0] + [i[0] for i in self._interfacesn] + [len(self._zn) - 1]]

        # Draw a vertical line and label for each interface
        for ii, i in enumerate(ipoints):
            if ii in [0, len(ipoints) - 1]:
                mpl.vlines(i, -.5, .25, linestyles='dashed')
            else:
                mpl.vlines(i, -.5, .25)
            mpl.text(i, .25, "{:.3g}".format(i), clip_on=True,
                     horizontalalignment='center', verticalalignment='bottom')

        # Draw a material label over each region
        for i, m in zip((ipoints[1:] + ipoints[:-1]) / 2, [l.name for l in self._layers]):
            mpl.text(i, .1, m, clip_on=True, horizontalalignment='center')

        # Draw a small vertical line for each mesh point
        mpl.vlines(self._zn, -.05, .05)

        # Fit the xlimits to the mesh
        if xlim is None:
            mpl.xlim(self._zn[0], self._zn[-1] + .1)
        else:
            mpl.xlim(xlim)

        # Fit the ylimits to the drawn lines
        mpl.ylim(-.05, .5)

        # Get rid of extra axes and ticks
        mpl.gca().get_yaxis().set_visible(False)
        mpl.gca().spines['top'].set_visible(False)
        mpl.gca().spines['right'].set_visible(False)
        mpl.gca().spines['left'].set_visible(False)
        mpl.gca().get_xaxis().tick_bottom()
        mpl.gca().get_yaxis().tick_left()

        mpl.title('Total mesh points: {:d}'.format(len(self._zn)))
        mpl.tight_layout()

    def _fill_from_matblocks(self,key,default=None):

        # If we have a default, build the function
        if default is not None:
            func=self[key]=MidFunction(self,value=default)
        # Otherwise MaterialBlock.get will build it the first time
        func=None

        # Check each material block
        for mb in self._matblocks:
            try:
                func=mb.get(key,destmesh=self,destfunc=func)
            except Exception as e:
                print(e)
                if default is None:
                    raise Exception("No default specified and {} not in {}".format(key,mb.matsys.name))
        self[key]=func
        return self[key]

    def globalize(self, func, default=None, submeshes=None):
        """ Finds the function `func` on submeshes and expands it to a function on the full mesh, ensuring that
        the functions on submeshes are just restricted views of the full mesh function (no longer independent).

        If the function is defined on multiple submeshes in incompatible ways (ie different shapes) or a `default` is
        supplied which is incompatible with the way this function is defined on some submesh, the results are not
        defined and errors may be raised.

        Args:
            func: (str) the function name to look for
            default: a default value to fill into the function where not defined on a submesh
        Returns:
            the function
        """
        if func in self._functions: return self[func]
        log("Expanding function "+func,level="debug")
        if submeshes is None:
            submeshes=self._submeshes

        # Find the first submesh to have this function and copy out the position/shape/dtype to full mesh
        for sm in submeshes:
            foundit=False
            if func in sm:
                sfunc=sm.globalize(func,default=default)
                if not foundit:
                    # Get the shape from the default if supplied
                    if default is not None:
                        self._functions[func]=Function(self,sfunc.pos,value=default,dtype=sfunc.dtype)
                    else:
                        self._functions[func]=Function(self,sfunc.pos,dtype=sfunc.dtype,empty=sfunc.shape[:-1])
                foundit=True

        # Fill the function on the entire mesh from anywhere it appears in submeshes
        for sm in submeshes:
            if func in sm:
                if sfunc.pos=='node':
                    self[func][...,sm._slicen]=sm[func]
                else:
                    self[func][...,sm._slicem]=sm[func]

                # Make submeshes a restricted view of the full mesh
                del sm._functions[func]
                sm[func]=self[func].restrict(sm)
        return self[func]

    def ensure_function_exists(self,func,value=np.nan,dim=(),pos='node',dtype='float'):
        """ If it doesn't exist, make it in the global mesh, if it does, check the dim/pos.

        Propagates the function upward to supermeshes.

        Args:
            fund: name of the function
            value,dim,pos,dtype: passed to :class:`Function`

        Returns:
            None
        """
        if self.__contains__(func):
            if not list(self[func].shape[:-1])==list(dim):
                raise Exception(func+" is the wrong shape: "+\
                    str(dim)+" requested "+str(self[func].shape[:-1])+" present.")
            if not self[func].pos==pos:
                raise Exception(func+" is the wrong mesh-type: "+\
                    pos + " requested "+str(self[func].pos+" present."))
        else:
            if self._supermesh==None:
                self[func]=Function(self,pos=pos,empty=dim,dtype=dtype, value=value)
                self[func][:]=value
            else:
                self._supermesh.ensure_function_exists(func,dim=dim,pos=pos,dtype=dtype,value=value)


    def __contains__(self,key):
        r""" True iff there is a function ``key`` defined on this mesh."""
        return (key in self._functions) or bool(sum([key in sm for sm in self._submeshes]))

    def __iter__(self):
        r""" Iterate through functions defined on this mesh."""
        return self._functions.__iter__()

    def __getitem__(self, key):
        return self.get(key)

    def get(self,key):
        r""" Get by name a function defined on this mesh.

        Search order: (1) if the function is already defined on this mesh, return it. (2) If any material block seems
        to have the variable, ask all material blocks to fill it in.  If they can't ALL fill it in, an exception will
        be raised.

        Note: if the function is defined on submeshes but not on the global mesh, this method will not find it, but
        you can call py:func:`pynitride.mesh.globalize` to bring it onto the global mesh.

        """
        #print("in get",key)
        if key in self._functions:
            return self._functions[key]
        elif sum(key in mb for mb in self._matblocks):
            return self._fill_from_matblocks(key)
        else:
            raise Exception("Trouble finding: "+key)

    def __setitem__(self, key, value):
        r""" Update (or create) a function on this mesh.  Propagates to any submeshes."""
        if key in self._functions:
            self._functions[key][:] = value
        else:
            assert isinstance(value,Function), "Must be a mesh.functions.Function"
            self._functions[key] = value
            def submeshesview(m,value):
                if not len(m._submeshes): return
                for sm in m._submeshes:
                    vres=value.restrict(sm)
                    sm._functions[key]=vres
                    submeshesview(sm,vres)
            submeshesview(self,value)

    def __getattr__(self,item):
        return self.__getitem__(item)

    def __setattr__(self,key,value):
        #print('in setattr',key)
        if ('_functions' in self.__dict__) and (key in self._functions):
            self.__setitem__(key,value)
        else:
            super().__setattr__(key,value)

    @property
    def zn(self):
        r""" the locations of the node mesh as a numpy array"""
        return self._zn

    @property
    def zm(self):
        r""" the locations of the mid mesh as a numpy array"""
        return self._zm

    @property
    def dzn(self):
        r""" the spacing between locations in the node mesh as a numpy array"""
        return self._dzn

    @property
    def dzm(self):
        r""" the spacing between locations in the mid mesh as a numpy array"""
        return self._dzm

    @property
    def interfaces_node(self):
        r""" list of interfaces and adjacent :class:`Layer`'s on the node mesh.

        :return: each element is a tuple of the form ``(index, layer_to_left, layer_to_right)``
        """
        return self._interfacesn
    @property
    def interfaces_mid(self):
        r""" list of interfaces and adjacent :class:`Layer`'s on the mid mesh.

        :return: each element is a tuple of the form
            ``(index_to_left, index_to_right, layer_to_left, layer_to_right)``
        """
        return self._interfacesm

    def submesh(self, zbounds, name="Temporary"):
        r""" Returns a :class:`SubMesh` viewing a range of this mesh.

        The range is specified by desired :math:`z` locations.  If you want to specify exact indices instead, use the
        :class:`SubMesh` constructor directly.

        :param zbounds: two-element tuple of :math:`z` locations to start and stop the submesh (inclusive)
        :return: a :class:`SubMesh` which views this mesh in the desired range
        """
        return SubMesh(self, name, self.indexn(zbounds[0]), self.indexn(zbounds[1]) + 1)

    def submesh_cover(self,znoints,names):
        """ Returns a cover of the mesh split into submeshes at each point

        Args:
            znoints: list of locations (nearest point in node mesh will be used) where to divide the meshes
            names: a list of names (length one greater than `znoints`)

        Returns:
            a list of submeshes

        """
        inds=[0]+self.indexn(znoints).tolist() + [len(self.zn) - 1]
        assert np.all(np.diff(inds)>0), "Zero-size or overlapping submeshes in cover"
        sms=[]
        for il,ir,name in zip(inds[:-1],inds[1:],names):
            sms+=[SubMesh(self,name,il,ir+1)]
        return sms

    def get_globalmesh(self):
        """ Returns the global mesh to which this submesh belongs directly or indirectly."""
        if self._supermesh is None: return self
        else: return self._supermesh.get_globalmesh()

    def has_submesh(self, submesh):
        """ Looks for submesh

        If `submesh` is a direct child of this mesh, returns True.
        If `submesh` is a child of this mesh somewhere down the line,
        returns the direct child submesh of this mesh under which `submesh` can be found.
        Otherwise returns False.
        """
        if not len(self._submeshes): return False
        if submesh in self._submeshes: return True 
        for sm in self._submeshes:
            if sm.has_submesh(submesh):
                return sm
        return False

    def save(self,filename,keys=None):
        """ Saves the mesh functions to a file (a numpy .npz)

        Args:
            filename: path at which to save
            keys: if provided, restrict the saved keys to only these
        """
        if keys is None:
            res=self._functions
        else:
            res={k:self[k] for k in keys}
        if filename:
            np.savez_compressed(filename,**res)
        else: return deepcopy(res)

    def read(self,filename):
        """ Reads the mesh functions from a file (a numpy .npz)

        Mesh should match the one saved from.

        Args:
            filename: path at which to save
        """
        with np.load(filename) as data:
            self.restore(data)

    def restore(self,data):
        for k,v in data.items():
            if v.shape[-1]==len(self._zn):
                self[k]=NodFunction(self,v)
            elif v.shape[-1]==len(self._zm):
                self[k]=MidFunction(self,v)
            else:
                raise Exception(k+" has the wrong shape "+str(v.shape)+" for this mesh.")

class SubMesh(Mesh):

    def __init__(self, mesh, name, start, stop):
        """ Represents a Mesh restricted to to particular segment of a larger mesh.

        A submesh is essentially just a view of a mesh within that segment, ie it has all of the same functionality as the
        :py:class:`~pynitride.poissolve.mesh.Mesh` from which it is created, except that all operations on it (eg setting
        or getting mesh functions) will only affect those functions within the prescribed range.  Vitally, the data is
        *shared*, not copied, between a mesh and its submeshes.

        Note: the constructor for SubMesh takes exact *indices* as bounds of the submesh region.  If you wish to instead specify
        :math:`z`-positions, see the :py:func:`Mesh.submesh` function

        Args:
            mesh: the super :py:class:`~pynitride.poissolve.mesh.Mesh` within which this submesh will reside
            start: The lower index (inclusive) of the submesh in the larger mesh
            stop: The upper index (exclusive) of the submesh in the larger mesh
        """
        self._supermesh = mesh
        self._submeshes = []
        if self not in mesh._submeshes:
            mesh._submeshes += [self]
        self.name=name

        if start is None: start=0
        if stop is None: stop=len(mesh.zn)

        self._slicen = slice(start, stop)
        self._slicem = slice(start, stop - 1)
        self._global_slicen=slice(mesh._global_slicen.start+start,mesh._global_slicen.start+stop)
        self._global_slicem=slice(mesh._global_slicem.start+start,mesh._global_slicem.start+stop-1)

        self._zn = mesh._zn[self._slicen]
        self._zm = mesh._zm[self._slicem]
        self._dzn = mesh._dzn[self._slicem]
        self._dzm = mesh._dzm[self._slicen]

        self._interfacesn = [(i -start,           ll, lr) for i,     ll, lr in mesh.interfaces_node if (i  > start and i  < stop - 1)]
        self._interfacesm = [(il-start, ir-start, ll, lr) for il,ir, ll, lr in mesh.interfaces_mid   if (il > start and ir < stop - 1)]
        # THIS IS A HORRIBLE HACK.  I'M SORRY, FUTURE SAM.
        if len(self.interfaces_node):
            self._layers = [ll for i, ll, lr in self.interfaces_node] + [self.interfaces_node[-1][2]]
        else:
            self._layers=[next(ll for i,ll,lr in (mesh.interfaces_node+[[start+1,mesh._layers[-1],None]]) if i > start)]

        self._functions = { k: f.restrict(self) for k, f in mesh._functions.items()}
        self._requested_functions = {}


        if len(self._zm)>1:
            # interpolate the zn -> index mapping
            self._zn2i_interp = interp1d(self._zn, np.arange(len(self._zn)))
            # interpolate the zm -> index mapping
            self._zm2i_interp = interp1d(self._zm, np.arange(len(self._zm)))

        self._matblocks=list(set(l._matblock for l in self._layers))
        self.ztrans=mesh.ztrans

        self.Nn=len(self._zn)
        self.Nm=len(self._zm)

        self.zeros_nod=NodFunction(self,0)
        self.zeros_mid=MidFunction(self,0)
        self.ones_nod=NodFunction(self,1)
        self.ones_mid=MidFunction(self,1)

        self._metric=assemble_load_matrix(self.ones_mid,self.dzn,n=1,dirichelet1=False,dirichelet2=False)


class Function(np.ndarray):
    r""" Represents a generic function defined on a :py:class:`~pynitride.poissolve.mesh.Mesh`.

    This classes subclasses Numpy's
    `ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_ so vectorized math should
    just work.  Functions are arbitrarily shaped/dimensioned ``ndarray``'s, with the constraint that the last dimension
    matches the length of the (point or mid) mesh on which it is defined.

    A Function can be initialized via the ``value`` or ``empty`` parameter in numerous ways:

        - if the ``empty`` argument is supplied, an empty (ie values uninitialized) array will be constructed with the
          shape specified as a tuple to ``empty``
        - if ``value`` is an array whose last dimension matches the mesh already, this function will just be a
          view of ``value``.
        - if ``value`` is a number (or generally, a non-iterable), it will be repeated to the length of the mesh.
        - if ``value`` is one-dimensional but doesn't match the mesh, it will be converted to two-dimensional and
          tiled along the trival axis to the length of the node mesh.

    Args:
        mesh: the :py:class:`~pynitride.poissolve.mesh.Mesh` on which this function is defined
        value: (see above) the default value to initialize this array with
        dtype: the Numpy data type of the array (See
            `ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_.)
        empty: (see above) False to use ``value``, or shape tuple to construct an array.

    """
    def __new__(cls, mesh, pos, value=np.nan, dtype='float', empty=False):
        if pos=='node': z=mesh.zn
        if pos=='mid' : z=mesh.zm

        # If the user just wants an empty array, the shape of an element is specified by empty
        if empty is not False:
            vshape=list(empty)
            obj = np.empty(vshape + list(z.shape), dtype=dtype).view(cls)
            obj.mesh=mesh
            obj.z=z
            obj.pos=pos
            return obj

        # Otherwise, read the shape from the given value
        value = np.asarray(value,dtype=dtype)
        vshape=list(value.shape)

        # If the shape matches up to the mesh already, go ahead and just view that value as the NodFunction
        if hasattr(z,'shape') and len(value.shape) and value.shape[-1]==z.shape[0]:
            obj=value.view(cls)
            obj.mesh = mesh
            obj.z=z
            obj.pos=pos
            return obj

        # If the shape is one dimensional (but doesn't match the mesh, see above), reshape it to 2d
        # with the length=1 along the last axis.
        elif len(value.shape)==1:
            value=np.reshape(value,(len(value),1))

        # Try to form a full array by repeating this element len(z) times.
        try:

            if not hasattr(z,'shape'):
                obj=value.view(cls)
            else:
                obj = np.full(vshape + list(z.shape), value, dtype=dtype).view(cls)
            obj.mesh = mesh
            obj.z=z
            obj.pos=pos
            return obj
        except:
            raise Exception("Given arr of shape {} is not compatible with given mesh of size {}".format(value.shape, z.shape[0]))

    def plot(self,*args,**kwargs):
        r""" Convenience function to plot this function versus mesh positions via matplotlib.

        ``*args`` and ``**kwargs`` are passed directly to matplotlib's ``plot()``
        (See `Matplotlib docs <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_.)
        """
        mpl.plot(self.z, self, *args, **kwargs)

    def __array_finalize__(self, obj):
        r""" Make sure that the ``mesh`` property is kept whenever a new Function is casted from another.

        See :ref:`Subclassing ndarray <https://docs.scipy.org/doc/numpy/user/basics.subclassing.html>`.

        Args:
            obj: the ndarray from which this array is being created
        """
        if obj is not None:
            self.mesh = getattr(obj, 'mesh', "View casting from ndarray not supported.")
            self.z =    getattr(obj, 'z'   , "View casting from ndarray not supported.")
            self.pos =  getattr(obj, 'pos' , "View casting from ndarray not supported.")

    def differentiate(self,fill_value=np.nan):
        r""" Central-difference derivative

        Differentiate, accounting for the appropriate potentially non-uniform mesh spacing.
        Note that the derivative of a node function is a mid function, and vice-versa.

        Args:
            fill_value: when differentiating a mid-function to a node-function, the central-difference derivative
                at the boundary is not defined, this parameter provides a way to fill those boundary points in.
        Returns:
             a Function representing the derivative.
        """
        if self.pos=='node':
            return Function(self.mesh, 'mid',np.diff(self, axis=-1) / self.mesh.dzn, dtype=self.dtype)
        if self.pos=='mid':
            pf = Function(self.mesh,'node',empty=np.array(self.T[0].shape),dtype=self.dtype)
            pf.T[1:-1] = (np.diff(self,axis=-1) / self.mesh.dzm[1:-1]).T
            pf.T[[0, -1]] = fill_value
            return pf


    def integrate(self,flipped=False,definite=False):
        r""" Cumulative integral in either direction, or definite integral

        Integrate, accounting for the appropriate potentially non-uniform mesh spacing
        Note that the integral of a node function is a mid function, and vice-versa.
        When integrating a node function to a mid function, the last point is ignored.
        When integrating a mid function to a node function, the first point is zero.

        Args:
            flipped: If True, integrate from :math:`+z` to :math:`-z`,
                rather than the default :math:`-z` to :math:`+z`
            definite: give just the total integral rather than computing the cumulative.  Note that, when integrating
                a mid function, this the last point of the cumulative result, but that's not true for a node function.
        Returns:
            a Function or, if ``definite``, just a number
        """
        if self.pos=='node':
            if definite:
                return np.sum(self * self.mesh.dzm,axis=-1)
            else:
                return Function(self.mesh,pos='mid',value=np.cumsum(
                    (self * self.mesh.dzm).T[:-1].T
                    if not flipped
                    else np.flipud(-self * self.mesh.dzm).T[:-1].T,
                    axis=-1))#.view(Function)
        if self.pos=="mid":
            if definite:
                return np.sum(self * self.mesh.dzn, axis=-1)
            else:
                output = Function(self.mesh,pos='node', value=0.0)
                np.cumsum(
                    self * self.mesh.dzn if not flipped else np.flipud(-self * self.mesh.dzn),
                    out=(output[1:] if not flipped else np.flipud(output[:-1])), axis=-1)
                return output

    def restrict(self, submesh):
        r""" Returns a view of this function restricted to the given submesh.

        Args:
            submesh: the submesh on which this function is needed
        Returns:
             a Function of the same type, which shares, not copies, data with the original
        """
        if submesh==self.mesh:
            return self
        seek=self.mesh.has_submesh(submesh)
        assert seek,\
            "Want to restrict but {} is not a submesh of {}"\
                .format(submesh,self.mesh)
        if seek is not True: submesh=seek
        if self.pos=='node':
            return (type(self)(submesh, pos='node',value=self.T[submesh._slicen].T,dtype=self.dtype)).restrict(submesh)
        if self.pos=='mid':
            return (type(self)(submesh, pos='mid' ,value=self.T[submesh._slicem].T,dtype=self.dtype)).restrict(submesh)

    def tpf(self, interp='z'):
        warnings.warn("Function.tpf() is deprecated, use Function.tnf().",DeprecationWarning)
        return self.tnf(interp=interp)

    def tnf(self, interp='z'):
        r""" Ensure that a Function is defined on the node mesh.

        Args:
            interp: ``'z'`` for a linear interpolation which accounts for non-uniform point spacing (and boundaries
                by extrapolation).
                ``'unweighted'`` for a straightforward average of adjacent points (and boundaries are the boundaries
                of the mid mesh).
        Returns:
            a node function (which is the original if its already a node function)
        """
        if self.pos=='node': return self
        if not hasattr(self.z,"shape"): return Function(self.mesh,pos='node',value=self,dtype=self.dtype)

        if len(self)==1: interp='unweighted'
        if interp == 'unweighted':
            newshape=list(self.shape)
            newshape[-1]+=1
            arr = np.empty(newshape,dtype=self.dtype).T
            arr[1:-1] = (self.T[1:] + self.T[:-1]) / 2
            arr[[0, -1]] = self.T[[0, -1]]
            arr=arr.T
        if interp == 'z':
            arr = interp1d(self.mesh.zm, self,
                           fill_value='extrapolate')(self.mesh.zn)
        return Function(self.mesh,pos='node',value=arr,dtype=arr.dtype)

    def tmf(self, interp='z'):
        r""" Ensure that a Function is defined on the node mesh.

        Args:
            interp: ``'z'`` for a linear interpolation which accounts for non-uniform point spacing (and boundaries
                by extrapolation).
                ``'unweighted'`` for a straightforward average of adjacent points (and boundaries are the boundaries
                of the mid mesh).
        Returns:
             a node function (which is the original if its already a node function)
        """
        if self.pos=='mid': return self

        if interp == 'unweighted':
            newshape=list(self.shape)
            newshape[-1]-=1
            arr = np.empty(newshape,dtype=self.dtype).T
            arr = (self.T[1:] + self.T[:-1]) / 2
            arr=arr.T
        if interp == 'z':
            arr = interp1d(self.mesh.zn, self,
                           fill_value='extrapolate')(self.mesh.zm)
        return Function(self.mesh,pos='mid',value=arr,dtype=self.dtype)

def NodFunction(mesh,value=np.nan,dtype='float',empty=False):
    r""" Returns a function defined on the node mesh

    This is a convenience equivalent to calling :py:func:`~pynitride.poissolve.mesh.Function` with ``pos='node'``.
    All other arguments are the same.
    """
    return Function(mesh,pos='node',value=value,dtype=dtype,empty=empty)
NodFunction=NodFunction

def MidFunction(mesh,value=np.nan,dtype='float',empty=False):
    r""" Returns a function defined on the mid mesh

    This is a convenience equivalent to calling :py:class:`~pynitride.poissolve.mesh.Function` with ``pos='mid'``.
    All other arguments are the same.
    """
    return Function(mesh,pos='mid',value=value,dtype=dtype,empty=empty)

def MaterialFunction(mesh, prop, default=None,dtype='float',pos='mid'):
    r""" Creates a Function by a piecewise-constant-over-materials definition.

    Args:
        mesh: the mesh on which the function is defined.
        prop: either (1) a function which will be called separately for each layer with the relevant
            :py:class:`pynitride.paramdb.Material` as the sole argument or (2) a property key which
            will be requested from each dictionary eg `"electron.mdos"`.
        default,pos: passed to :func:`MidFunction`
    Return:
        a :func:`MidFunction`
    """

    if default is not None:
        func=MidFunction(mesh,value=default,dtype=dtype)
    else:
        func=MidFunction(mesh,dtype=dtype,empty=())

    for mb in mesh._matblocks:
        try:
            mb.get(prop,destfunc=func)
        except Exception as e:
            if default is None:
                raise Exception("Could not get \"{}\" from MaterialBlock \"{}\" and no default supplied".format(prop,mb.name))

    if pos=='mid':
        return func
    else:
        return func.tpf()

def LinearFunction(mesh, vstart, vstop, pos='node'):
    r""" A function which is only non-zero at a single location

    Args:
        mesh: the mesh on which this function is defined
        vstart: the value at the start (first z) of the function
        vstop: the value at the end (last z) of the function
        pos: build on a node mesh or mid mesh
    Returns:
        a linear function from ``vstart`` to ``vstop`` as a :py:class:`Function`
    """
    z={'node': mesh.zn, 'mid': mesh.zm}[pos]
    func={'node': NodFunction, 'mid': MidFunction}[pos](mesh,
        (vstop-vstart)*(z-z[0])/(z[-1]-z[0])+vstart)
    return func

def DeltaFunction(mesh, z, integral=1, i=None, pos='node'):
    r""" A function which is only non-zero at a single location

    Args:
        mesh: the mesh on which this function is defined
        z: the location of the delta function (index nearest this point will be used)
        integral: the integrated value of this delta function (ie prefactor)
        i: if specified, ``z`` will be ignored and ``i`` is the exact index where the delta will be placed
        pos: build on a node mesh or mid mesh
    Returns:
        the delta function as a :py:class:`Function`
    """
    func={'node': NodFunction, 'mid': MidFunction}[pos](mesh,0.0)
    i={'node': mesh.indexn, 'mid': mesh.indexm}[pos](z) if i is None else i
    func[i]= integral / {'node':mesh.dzm[i], 'mid':mesh.dzn[i]}[pos]
    return func

def inner_product(a,b):
    """ Takes the FEM inner product between two functions.

    Args:
        a,b:  the two :func:`NodFunction`

    Returns:
        the inner product with respect to the FEM metric

    """
    assert a.mesh is b.mesh
    assert a.pos=='node'
    assert b.pos=='node'

    metric=a.mesh._metric

    return np.sum(a.conj().T*(metric @ b.T))

