# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:41:30 2017

@author: sam
"""
import numpy as np
import matplotlib.pyplot as mpl
from scipy.interpolate import interp1d
import numpy as np


class Function(np.ndarray):
    def __new__(cls, *args, **kwargs):
        raise NotImplementedError

    def __array_finalize__(self, obj):
        if obj is None: return

        self.mesh = getattr(obj, 'mesh', "View casting from ndarray not supported.")

    def restrict(self, submesh):
        # doesn't check that submesh and mesh are compatible
        return type(self)(submesh, self[submesh._slice])


class PointFunction(Function):
    def __new__(cls, mesh, value=np.NaN, dtype='float'):
        if hasattr(value, '__iter__'):
            obj = np.asarray(value).view(cls)
            assert obj.shape[-1] == mesh.z.shape[-1], \
                "Given arr of shape {} is not compatible with given mesh of size {}".format(obj.shape,mesh.z.shape[0])
        else:
            obj = np.full(mesh.z.shape, value, dtype=dtype).view(cls)
        obj.mesh = mesh
        return obj

    def plot(self,*args,**kwargs):
        mpl.plot(self.mesh.z,self,*args,**kwargs)

    # def __array_prepare__(self, out_arr, context=None):
    #    assert out_arr.shape==self.mesh.z.shape,\
    #        "Can't combine Functions of different mesh sizes"
    #    out_arr.shape=self.mesh.z.shape
    #    return out_arr

    def differentiate(self):
        return MidFunction(self.mesh, np.diff(self, axis=-1) / self.mesh._dz)

    # provide a non-cumsum, just sum, version for efficiency when that's all that's wanted
    def integrate(self, flipped=False):
        return np.cumsum(
            (self * self.mesh._dzp).T[:-1].T
            if not flipped
            else np.flipud(-self * self.mesh._dzp).T[:-1].T,
            axis=-1).view(MidFunction)


class MidFunction(Function):
    def __new__(cls, mesh, value=np.NaN, dtype='float'):
        if hasattr(value, '__iter__'):
            obj = np.asarray(value).view(cls)
            assert obj.shape[-1] == mesh.zp.shape[-1], \
                "Given arr is not compatible with given mesh"
        else:
            obj = np.full(mesh.zp.shape, value, dtype=dtype).view(cls)
        obj.mesh = mesh
        return obj

    def plot(self,*args,**kwargs):
        mpl.plot(self.mesh.zp,self,*args,**kwargs)
        # def __array_prepare__(self, out_arr, context=None):
    #    assert out_arr.shape==self.mesh.zp.shape,\
    #        "Can't combine Functions of different mesh sizes"
    #    out_arr.shape=self.mesh.zp.shape
    #    return out_arr

    def differentiate(self, fill_value=np.NaN):
        pf = PointFunction(self.mesh)
        pf[1:-1] = np.diff(self,axis=-1) / self.mesh._dzp[1:-1]
        pf[[0, -1]] = fill_value
        return pf

    def integrate(self, flipped=False):
        # if output is None:
        output = PointFunction(self.mesh, value=0.0)
        np.cumsum(
            self * self.mesh._dz if not flipped else np.flipud(-self * self.mesh._dz),
            out=(output[1:] if not flipped else np.flipud(output[:-1])), axis=-1)
        return output

    def to_point_function(self, interp='unweighted'):
        if interp == 'unweighted':
            newshape=list(self.shape)
            newshape[-1]+=1
            arr = np.empty(newshape).T
            arr[1:-1] = (self.T[1:] + self.T[:-1]) / 2
            arr[[0, -1]] = arr[[1, -2]]
            arr=arr.T
        if interp == 'z':
            arr = interp1d(self.mesh.zp, self,
                           fill_value='extrapolate')(self.mesh.z)
        return PointFunction(self.mesh, arr)

def ConstantFunction(mesh, val, dtype='float', pos='point'):
    from numpy.lib.stride_tricks import as_strided
    x = np.array(val, order='C', dtype=dtype)
    newshape = list(x.shape) + [mesh.z.shape[0] if pos=='point' else mesh.zp.shape[0]]
    newstrides = list(x.strides) + [0]
    arr=as_strided(np.array(x), shape=newshape, strides=newstrides)
    return {'point': PointFunction, 'mid': MidFunction}[pos](mesh,value=arr)

def MaterialFunction(mesh, prop, pos='mid'):
    # could make this more efficient by directly interpolating if Point case?
    # this function almost duplicates RegionFunction...

    ptcounts = np.diff([0] + [i for i, ll, lr in mesh.interfaces_point] + [len(mesh.z) - 1])
    arr = []

    propfunc = (lambda i: prop(mesh._layers[i].material)) \
        if callable(prop) \
        else (lambda i: mesh._layers[i][prop])
    for i, ptc in enumerate(ptcounts):
        arr += [propfunc(i)] * ptc

    out = MidFunction(mesh, np.array(arr).T)
    if pos == "point":
        return out.to_point_function()
    else:
        return out


def RegionFunction(mesh, prop, pos='mid'):
    # could make this more efficient by directly interpolating if Point case?

    ptcounts = np.diff([0] + [i for i, ll, lr in mesh.interfaces_point] + [len(mesh.z) - 1])
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
    i={'point': mesh.index, 'mid': mesh.indexp}[pos](z) if i is None else i
    func[i]=height/{'point':mesh._dzp[i], 'mid':mesh._dz[i]}[pos]
    return func

if __name__=="__main__":
    from runpy import run_path
    run_path('tests/test_functions.py',run_name='__main__')
    from poissolve.tests.test_functions import *