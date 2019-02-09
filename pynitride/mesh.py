# -*- coding: utf-8 -*-
r""" Meshing, submeshing, and manipulating functions defined on meshes.

This module is tested by :py:mod:`~tests.test_mesh`.
"""

import matplotlib.pyplot as mpl
import numpy as np
from matplotlib import pyplot as mpl
from scipy.interpolate import interp1d
from math import gcd,ceil
from functools import reduce
from pynitride.visual import log
from scipy.special import lambertw as W
from pynitride.fem import assemble_load_matrix

class MaterialBlock():
    def __init__(self,name,matsys,layers):
        self.name=name
        self.matsys=matsys
        self.layers=layers
        for l in layers:
            l._matblock=self

    def place(self,mesh):
        self._mesh=mesh
        mesh.name=self.name

    @property
    def mesh(self):
        return self._mesh

    def get(self, item, destmesh=None, destfunc=None):
        if destfunc is not None:
            destmesh=destfunc.mesh
        elif destmesh is None:
            destmesh=self._mesh

        #print('item ',item,'   destmesh', destmesh.zp.shape)

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
            #print("is already defined")
            func=destmesh[item]

        # Or if we haven't made the global func yet, make it from this one
        elif destfunc is None:
            func=MidFunction(destmesh, dtype=subfunc.dtype, empty=subfunc.shape[:-1])
            #print("defining now")
        else:
            func=destfunc

        # Figure out the ranges where the desination mesh and material block overlap
        globalstart=max(self._mesh._global_slicem.start,destmesh._global_slicem.start)
        globalstop =min(self._mesh._global_slicem.stop ,destmesh._global_slicem.stop )


        #print("ITEM: ",item,"  matblock ",self.name,"  ",self.mesh.zm.shape," destmesh  ", destmesh.zm.shape)
        #print("    ",globalstart,"  -  ",globalstop)
        #print(globalstart-destmesh._global_slicem.start,globalstop-destmesh._global_slicem.start)
        #print(globalstart-self._mesh._global_slicem.start,globalstop-self._mesh._global_slicem.start)
        #print(subfunc[globalstart-self._mesh._global_slicem.start:globalstop-self._mesh._global_slicem.start].shape)
        #print("are there nans in subfunc?",np.isnan(subfunc))
        # Fill in the relevant part of the function
        func[globalstart-destmesh._global_slicem.start:globalstop-destmesh._global_slicem.start]=\
            subfunc.T[globalstart-self._mesh._global_slicem.start:globalstop-self._mesh._global_slicem.start].T
        #print('hi')
        return func

    def update(self,reason,destmesh):
        for f in self.matsys._updates[reason]:
            f(destmesh)

    def __contains__(self, item):
        return (item in self.matsys) or (item in self._mesh._functions)

class Layer():
    def __init__(self, name, thickness):
        self.name = name
        self.thickness = thickness
    def place(self,mesh):
        self._mesh=mesh
        mesh.name=self.name

    @property
    def mesh(self):
        return self._mesh

    @property
    def matblock(self):
        return self._matblock

    def __getitem__(self, key):
        return self._mesh[key]

class UniformLayer(Layer):
    def __init__(self, name, thickness, **kwargs):
        super().__init__(name,thickness)
        self._setproperties=kwargs

    def place(self,mesh):
        super().place(mesh)
        for k,v in self._setproperties.items():
            if type(v) is bool:
                dtype='bool'
            else:
                dtype='float'
            mesh.create_restricted_function(k,MidFunction(mesh,value=v,dtype=dtype))


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

    def __init__(self, stack, max_dz, refinements=[], uniform=False, subs=None, boundary=["GenericMetal","thick"]):
        self._boundary=boundary
        self.ztrans=-1
        self._matblocks=stack
        self._layers = layers = sum([mb.layers for mb in stack],[])
        if subs is None:
            assert isinstance(layers[-1],UniformLayer),\
                "If no substrate explicitly specified, bottom layer must be uniform"
            self._subs=layers[-1]
        else:
            self._subs=stack[-1].matsys.bulk(**subs)

        #self._namedinterfacesz={}
        #for l in layers:


        # Parse refinements
        for r in refinements:
            zr=r[0]
            if isinstance(zr,str):
                l1,l2=zr.split("/")
                try:
                    l1,_=next((i,l) for i,l in enumerate(layers) if l.name==l1)
                    l2,_=next((i,l) for i,l in enumerate(layers) if l.name==l2)
                except:
                    raise Exception("A layer ({} or {}) was not found for refinement.".format(l1,l2))
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
            ## Implement the max_dz requirement by adding it to the refinements list
            #if refinements:
            #    refinements = np.vstack([np.array(refinements), [0, max_dz, 1]])
            #else:
            #    refinements = np.array([[0, max_dz, 1]])

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
        self._zp = np.array(fixed_positions)
        self._dzp = np.diff(self._zp)

        # Compile a list of interfaces for z
        interface_indices=np.array(interface_indices,dtype=int)
        # Each element is a tuple of the form (index, left layer, right layer)
        self._interfacesp = list(zip(interface_indices, layers[:-1], layers[1:]))
        # Compile a list of interfaces for zp
        # Each element is a tuple of the form (lindex,rindex, left layer, right layer)
        self._interfacesm= list(zip(interface_indices - 1, interface_indices, layers[:-1], layers[1:]))

        # Also keep the z's in-between mesh points
        self._zm = (self._zp[:-1] + self._zp[1:]) / 2
        self._dzm = np.array([self._dzp[0]] * len(self._zp))
        self._dzm[1:-1] = np.diff(self._zm)
        self._dzm[[0, -1]] = self._dzp[[0, -1]]/2

        if len(self._zm)>1:
            # interpolate the z -> index mapping
            self._zp2i_interp = interp1d(self._zp, np.arange(len(self._zp)))
            # interpolate the zp -> index mapping
            self._zm2i_interp = interp1d(self._zm, np.arange(len(self._zm)))

        # Store functions which live on this mesh
        self._functions = {}
        self._attrs = {}
        self._requested_functions = {}
        self._submeshes= []

        self._global_slicep=slice(0,len(self._zp))
        self._global_slicem=slice(0,len(self._zm))


        # TODO: break this up so some is done within matblock

        # This is the whole world
        self.name='global'
        self._supermesh = None
        leftindices=[0]+interface_indices.tolist()
        rightindices=interface_indices.tolist()+[len(self._zp)-1]
        ill=-1
        ilr=-1
        for i,mb in enumerate(self._matblocks):
            ill=ilr+1
            ilr=ill+len(mb.layers)-1
            ml=leftindices[ill]
            mr=rightindices[ilr]
            mb.place(SubMesh(self, '', ml, mr+1))
            ###
            for k,v in mb.matsys._defaults.items():
                mb.mesh.create_restricted_function(k,MidFunction(mb.mesh,v))
            ###
            for lay,l,r in zip(mb.layers,leftindices[ill:ilr+1],rightindices[ill:ilr+1]):
                lay.place(SubMesh(mb.mesh,'',l-ml,r-ml+1))


        self.Np=len(self._zp)
        self.Nm=len(self._zm)

        self.zeros_nod=NodFunction(self,0)
        self.zeros_mid=MidFunction(self,0)
        self.ones_nod=NodFunction(self,1)
        self.ones_mid=MidFunction(self,1)

        self._metric=assemble_load_matrix(self.ones_mid,self.dzp,n=1,dirichelet1=False,dirichelet2=False)

    def __repr__(self):
        return "<Mesh("+str(self.Np)+") \""+str(self.name)+"\">"

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

    def matblock(self,name):
        return next(mb for mb in self._matblocks if mb.name==name)

    def plot_mesh(self,xlim=None):
        """ Plots a 1-D representation of the mesh for visual inspection.
        """

        # Make a long, thin figure
        mpl.figure(figsize=(8, 2))

        # Collect the z values at interfaces
        ipoints = self._zp[[0] + [i[0] for i in self._interfacesp] + [len(self._zp) - 1]]

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
        mpl.vlines(self._zp, -.05, .05)

        # Fit the xlimits to the mesh
        if xlim is None:
            mpl.xlim(self._zp[0], self._zp[-1] + .1)
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

        mpl.title('Total mesh points: {:d}'.format(len(self._zp)))
        mpl.tight_layout()


    #def request_function(self,func,default=None,pos=None):
    #    self._requested_functions[func]={'default': default,'pos':pos}
    #def request_functions(self,funcs,defaults=[],poss=[]):
    #    if not len(defaults):
    #        defaults=[None]*len(funcs)
    #    if not len(poss):
    #        poss=[None]*len(funcs)
    #    assert len(funcs)==len(defaults) and len(funcs)==len(poss)
    #    for func,default,pos in zip(funcs,defaults,poss):
    #        self._requested_functions[func]={'default': default, 'pos': pos}

    #def initialize(self):
    #    """

    #    Before this function is called,
    #    (1) all exchanged non-material functions should be created (but maybe not globalized)
    #    (2) no material functions (on submeshes that are not on this mesh) should be created

    #    Go through the requested functions one by one and
    #    (1) if the function exists on this mesh, do nothing
    #    (2) if the function exists on a submesh (note it is not a material function), globalize it
    #    (3) if the function can be drawn from all the relevant material blocks (and/or there is a default value), do so

    #    """

    #    for key,v in self._requested_functions.items():
    #        if key in self._functions:
    #            continue
    #        else:
    #            default,pos=v['default'],v['pos']
    #            if sum([key in sm for sm in self._submeshes]):
    #                self.globalize(key,default=default)
    #            else:
    #                if pos=='point':
    #                    assert default is not None,"Specified pos=='point' for key "+key+"but no default"
    #                    self[key]=PointFunction(self,value=default)
    #                else:
    #                    self._fill_from_matblocks(key,default)

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

        :param func: (str) the function name to look for
        :param default: a default value to fill into the function where not defined on a submesh
        :return: the function
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
                if sfunc.pos=='point':
                    self[func][...,sm._slicep]=sm[func]
                else:
                    self[func][...,sm._slicem]=sm[func]

                # Make submeshes a restricted view of the full mesh
                del sm._functions[func]
                sm[func]=self[func].restrict(sm)
        return self[func]

    def ensure_function_exists(self,func,value=np.NaN,dim=(),pos='point',dtype='float'):
        """ If it doesn't exist, make it in the global mesh, if it does, check the dim/pos.

        :param func:
        :param dim:
        :param pos:
        :param dtype:
        :param value:
        :return:
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

        Search order: (1) if the function is already defined on this mesh, return it.

        Note: if the function is defined on submeshes but not on the global mesh, this method will not find it, but
        you can call py:func:`pynitride.mesh.globalize` to bring it onto the global mesh.

        """
        if key in self._functions:
            return self._functions[key]
        elif key in self._attrs:
            self._attrs[key]()
            return self._functions[key]
        elif sum(key in mb for mb in self._matblocks):
            return self._fill_from_matblocks(key)
        else:
            raise Exception("EEH? "+key)

    def add_attr(self,attr,func):
        self._attrs[attr]=func
        for sm in self._submeshes:
            sm.add_attr(attr,func)

    def __setitem__(self, key, value, restricted=False):
        r""" Update (or create) a function on this mesh.  Propagates to any submeshes."""
        if key in self._functions:
            self._functions[key][:] = value
        else:
            #assert restricted or (self._supermesh==None), "Cannot set functions on submeshes"
            assert isinstance(value,Function), "Must be a mesh.functions.Function"
            self._functions[key] = value
            def submeshesview(m,value):
                if not len(m._submeshes): return
                for sm in m._submeshes:
                    vres=value.restrict(sm)
                    sm._functions[key]=vres
                    submeshesview(sm,vres)
            submeshesview(self,value)

    def create_restricted_function(self,key,value):
        self.__setitem__(key,value,restricted=True)

    def __getattr__(self,item):
        return self.__getitem__(item)

    @property
    def subs(self):
        return self._subs

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
        return SubMesh(self, name, self.indexp(zbounds[0]), self.indexp(zbounds[1]) + 1)

    def submesh_cover(self,zpoints,names):
        inds=[0]+self.indexp(zpoints).tolist()+[len(self.zp)-1]
        sms=[]
        for il,ir,name in zip(inds[:-1],inds[1:],names):
            sms+=[SubMesh(self,name,il,ir+1)]
        return sms

    def function_chart(self,submeshchain=[]):
        allfuncs=list(set(sum([list(m._functions.keys()) for m in [self]+submeshchain],[])))
        #print(allfuncs)
        def share(a,b):
            return (a.base is not None and a.base is b.base) or (a.base is b) or (b.base is a)
        table=[]
        for m,m2 in zip([self]+submeshchain, submeshchain+[None]):
            table+=[[f if f in m._functions else "" for f in allfuncs]]
            if m2:
                assert m2 in m._submeshes
                table+=[[(" -> " if ((f in m._functions)
                                     and (f in m2._functions)
                                     and share(m2._functions[f],m._functions[f]))
                                else "    ")\
                         for f in allfuncs]]
        table=zip(*table)
        print("---------------------------")
        for r in table:
            lout=""
            for i,c in enumerate(r):
                if (i+1) % 2:
                    lout+=c.rjust(30)
                else:
                    lout+=c
            print(lout)
        print("---------------------------")


    def save(self,filename,keys=None):
        if keys is None:
            res=self._functions
        else:
            res={k:self[k] for k in keys}
        np.savez_compressed(filename,**res)
    def read(self,filename):
        with np.load(filename) as data:
            for k,v in data.items():
                if v.shape[-1]==len(self._zp):
                    self[k]=PointFunction(self,v)
                elif v.shape[-1]==len(self._zm):
                    self[k]=MidFunction(self,v)
                else:
                    raise Exception(k+" has the wrong shape "+str(v.shape)+" for this mesh.")

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
    """ Represents a Mesh restricted to to particular segment of a larger mesh.

    A submesh is essentially just a view of a mesh within that segment, ie it has all of the same functionality as the
    :py:class:`~pynitride.poissolve.mesh.Mesh` from which it is created, except that all operations on it (eg setting
    or getting mesh functions) will only affect those functions within the prescribed range.  Vitally, the data is
    *shared*, not copied, between a mesh and its submeshes.

    Note: the constructor for SubMesh takes exact *indices* as bounds of the submesh region.  If you wish to instead specify
    :math:`z`-positions, see the :py:func:`~pynitride.poissolve.mesh.Mesh.submesh` function of
    :py:class:`~pynitride.poissolve.mesh.Mesh`.

    :param mesh: the super :py:class:`~pynitride.poissolve.mesh.Mesh` within which this submesh will reside
    :param start: The lower index (inclusive) of the submesh in the larger mesh
    :param stop: The upper index (exclusive) of the submesh in the larger mesh
    """

    """ Construct a submesh.

    Arguments:
        mesh: the Mesh from which to draw data
        start: the start index (inclusive) of the slice
        stop: the stop index (exclusive) of the slice

    """

    def __init__(self, mesh, name, start, stop):
        self._supermesh = mesh
        self._submeshes = []
        if self not in mesh._submeshes:
            mesh._submeshes += [self]
        self.name=name

        if start is None: start=0
        if stop is None: stop=len(mesh.zp)

        self._slicep = slice(start, stop)
        self._slicem = slice(start, stop - 1)
        self._global_slicep=slice(mesh._global_slicep.start+start,mesh._global_slicep.start+stop)
        self._global_slicem=slice(mesh._global_slicem.start+start,mesh._global_slicem.start+stop-1)

        self._zp = mesh._zp[self._slicep]
        self._zm = mesh._zm[self._slicem]
        self._dzp = mesh._dzp[self._slicem]
        self._dzm = mesh._dzm[self._slicep]

        self._interfacesp = [(i -start,           ll, lr) for i,     ll, lr in mesh.interfaces_point if (i  > start and i  < stop - 1)]
        self._interfacesm = [(il-start, ir-start, ll, lr) for il,ir, ll, lr in mesh.interfaces_mid   if (il > start and ir < stop - 1)]
        # THIS IS A HORRIBLE HACK.  I'M SORRY, FUTURE SAM.
        if len(self.interfaces_point):
            self._layers = [ll for i, ll, lr in self.interfaces_point] + [self.interfaces_point[-1][2]]
        else:
            self._layers=[next(ll for i,ll,lr in (mesh.interfaces_point+[[start+1,mesh._layers[-1],None]]) if i > start)]

        self._functions = { k: f.restrict(self) for k, f in mesh._functions.items()}
        self._attrs = mesh._attrs.copy()
        self._requested_functions = {}


        if len(self._zm)>1:
            # interpolate the zp -> index mapping
            self._zp2i_interp = interp1d(self._zp, np.arange(len(self._zp)))
            # interpolate the zm -> index mapping
            self._zm2i_interp = interp1d(self._zm, np.arange(len(self._zm)))

        self._matblocks=list(set(l._matblock for l in self._layers))
        self.ztrans=mesh.ztrans
        self._subs=mesh._subs

        self.Np=len(self._zp)
        self.Nm=len(self._zm)

        self.zeros_nod=NodFunction(self,0)
        self.zeros_mid=MidFunction(self,0)
        self.ones_nod=NodFunction(self,1)
        self.ones_mid=MidFunction(self,1)

        self._metric=assemble_load_matrix(self.ones_mid,self.dzp,n=1,dirichelet1=False,dirichelet2=False)


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
          tiled along the trival axis to the length of the point mesh.

    :param mesh: the :py:class:`~pynitride.poissolve.mesh.Mesh` on which this function is defined
    :param value: (see above) the default value to initialize this array with
    :param dtype: the Numpy data type of the array (See
        `ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_.)
    :param empty: (see above) False to use ``value``, or shape tuple to construct an array.

    """
    def __new__(cls, mesh, pos, value=np.NaN, dtype='float', empty=False):
        if pos=='point': z=mesh.zp
        if pos=='mid': z=mesh.zm

        # If the user just wants an empty array, the shape of an element is specified by empty
        if empty:
            vshape=list(empty)
            obj = np.empty(vshape + list(z.shape), dtype=dtype).view(cls)
            obj.mesh=mesh
            obj.z=z
            obj.pos=pos
            return obj

        # Otherwise, read the shape from the given value
        value = np.asarray(value,dtype=dtype)
        vshape=list(value.shape)

        # If the shape matches up to the mesh already, go ahead and just view that value as the PointFunction
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

        :param obj: the ndarray from which this array is being created
        """
        if obj is not None:
            self.mesh = getattr(obj, 'mesh', "View casting from ndarray not supported.")
            self.z =    getattr(obj, 'z'   , "View casting from ndarray not supported.")
            self.pos =  getattr(obj, 'pos' , "View casting from ndarray not supported.")

    def differentiate(self,fill_value=np.NaN):
        r""" Central-difference derivative

        Differentiate, accounting for the appropriate potentially non-uniform mesh spacing.
        Note that the derivative of a point function is a mid function, and vice-versa.

        :param fill_value: when differentiating a mid-function to a point-function, the central-difference derivative
            at the boundary is not defined, this parameter provides a way to fill those boundary points in.
        :return: a Function representing the derivative.
        """
        if self.pos=='point':
            return Function(self.mesh, 'mid',np.diff(self, axis=-1) / self.mesh.dzp, dtype=self.dtype)
        if self.pos=='mid':
            pf = Function(self.mesh,'point',empty=np.array(self.T[0].shape),dtype=self.dtype)
            pf.T[1:-1] = (np.diff(self,axis=-1) / self.mesh.dzm[1:-1]).T
            pf.T[[0, -1]] = fill_value
            return pf


    def integrate(self,flipped=False,definite=False):
        r""" Cumulative integral in either direction, or definite integral

        Integrate, accounting for the appropriate potentially non-uniform mesh spacing
        Note that the integral of a point function is a mid function, and vice-versa.
        When integrating a point function to a mid function, the last point is ignored.
        When integrating a mid function to a point function, the first point is zero.

        :param flipped: If True, integrate from :math:`+z` to :math:`-z`,
            rather than the default :math:`-z` to :math:`+z`
        :param definite: give just the total integral rather than computing the cumulative.  Note that, when integrating
            a mid function, this the last point of the cumulative result, but that's not true for a point function.
        :return: a Function or, if ``definite``, just a number
        """
        if self.pos=='point':
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
                return np.sum(self * self.mesh._dzp, axis=-1)
            else:
                output = Function(self.mesh,pos='point', value=0.0)
                np.cumsum(
                    self * self.mesh._dz if not flipped else np.flipud(-self * self.mesh._dz),
                    out=(output[1:] if not flipped else np.flipud(output[:-1])), axis=-1)
                return output

    def restrict(self, submesh):
        r""" Returns a view of this function restricted to the given submesh.

        :param submesh: the submesh on which this function is needed
        :return: a Function of the same type, which shares, not copies, data with the original
        """
        if submesh==self.mesh:
            return self
        assert submesh in self.mesh._submeshes,\
            "Haven't implemented recursive submeshing, going from {} to {}"\
                .format(self.mesh,submesh)
        if self.pos=='point':
            return type(self)(submesh, pos='point',value=self.T[submesh._slicep].T,dtype=self.dtype)
        if self.pos=='mid':
            return type(self)(submesh, pos='mid',value=self.T[submesh._slicem].T,dtype=self.dtype)

    def tpf(self, interp='z'):
        r""" Ensure that a Function is defined on the point mesh.

        :param interp: ``'z'`` for a linear interpolation which accounts for non-uniform point spacing (and boundaries
            by extrapolation).
            ``'unweighted'`` for a straightforward average of adjacent points (and boundaries are the boundaries
            of the mid mesh).
        :return: a point function (which is the original if its already a point function)
        """
        if self.pos=='point': return self
        if not hasattr(self.z,"shape"): return Function(self.mesh,pos='point',value=self,dtype=self.dtype)

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
                           fill_value='extrapolate')(self.mesh.zp)
        return Function(self.mesh,pos='point',value=arr,dtype=arr.dtype)

    def tmf(self, interp='z'):
        r""" Ensure that a Function is defined on the point mesh.

        :param interp: ``'z'`` for a linear interpolation which accounts for non-uniform point spacing (and boundaries
            by extrapolation).
            ``'unweighted'`` for a straightforward average of adjacent points (and boundaries are the boundaries
            of the mid mesh).
        :return: a point function (which is the original if its already a point function)
        """
        if self.pos=='mid': return self

        if interp == 'unweighted':
            newshape=list(self.shape)
            newshape[-1]-=1
            arr = np.empty(newshape,dtype=self.dtype).T
            arr = (self.T[1:] + self.T[:-1]) / 2
            arr=arr.T
        if interp == 'z':
            arr = interp1d(self.mesh.zp, self,
                           fill_value='extrapolate')(self.mesh.zm)
        return Function(self.mesh,pos='mid',value=arr,dtype=self.dtype)

def NodFunction(mesh,value=np.NaN,dtype='float',empty=False):
    r""" Returns a function defined on the point mesh

    This is a convenience equivalent to calling :py:func:`~pynitride.poissolve.mesh.Function` with ``pos='point'``.
    All other arguments are the same.
    """
    return Function(mesh,pos='point',value=value,dtype=dtype,empty=empty)
PointFunction=NodFunction

def MidFunction(mesh,value=np.NaN,dtype='float',empty=False):
    r""" Returns a function defined on the mid mesh

    This is a convenience equivalent to calling :py:class:`~pynitride.poissolve.mesh.Function` with ``pos='mid'``.
    All other arguments are the same.
    """
    return Function(mesh,pos='mid',value=value,dtype=dtype,empty=empty)

def ConstantFunction(*args,**kwargs):
    r""" Defines a function which is a single repeated constant.

    At the moment, this is just a do-nothing wrapper around :py:func:`~PointFunction`, but in the future this will provide a
    hook to implement operations which take advantage of the constancy.
    """
    return PointFunction(*args,**kwargs)
    #raise NotImplementedError

def MaterialFunction(mesh, prop, default=None,dtype='float',pos='mid'):
    r""" Creates a Function by a piecewise-constant-over-materials definition.

    :param mesh: the mesh on which the function is defined.
    :param prop: either (1) a function which will be called separately for each layer with the relevant
        :py:class:`pynitride.paramdb.Material` as the sole argument or (2) a property key which will be requested from each
        dictionary eg `"electron.mdos"`.
    :param default:
    :param pos:
    :return:
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
        return out.tpf()
    else:
        return out


def DeltaFunction(mesh, z, integral=1, i=None, pos='point'):
    r""" A function which is only non-zero at a single location

    :param mesh: the mesh on which this function is defined
    :param z: the location of the delta function (index nearest this point will be used)
    :param integral: the integrated value of this delta function (ie prefactor)
    :param i: if specified, ``z`` will be ignored and ``i`` is the exact index where the delta will be placed
    :param pos: build on a point mesh or mid mesh
    :return: the delta function as a :py:class:`Function`
    """
    func={'point': PointFunction, 'mid': MidFunction}[pos](mesh,0.0)
    i={'point': mesh.indexp, 'mid': mesh.indexp}[pos](z) if i is None else i
    func[i]= integral / {'point':mesh._dzp[i], 'mid':mesh.dzm[i]}[pos]
    return func

def inner_product(a,b):
    assert a.mesh is b.mesh
    assert a.pos=='point'
    assert b.pos=='point'

    metric=a.mesh._metric

    return np.sum(a.conj().T*(metric @ b.T))

