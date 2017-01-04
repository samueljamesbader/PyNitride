# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:41:30 2017

@author: sam
"""
import numpy as np
import matplotlib as mpl
from scipy.interpolate import interp1d


class Function():
    def __init__(self):
        raise NotImplementedError
    
    @property
    def array(self):
        return self._arr
    
    @array.setter
    def array(self,newarray):
        self._arr[:]=newarray
        
    def restrict(self,submesh):
        raise NotImplementedError
        
    def plot(self,*args,**kwargs):
        mpl.plot(self._z,self._arr,*args,**kwargs)
        
    

class PointFunction(Function):
    def __init__(self,mesh,arr=np.NaN):
        # doesn't check that mesh and arr are compatible
        self._mesh=mesh
        self._z=mesh.z
        if hasattr(arr,'__iter__'):
            self._arr=arr
        else:
            self._arr=np.array([arr]*len(self._z))

    def restrict(self,submesh):
        # doesn't check that submesh and mesh are compatible
        return PointFunction(submesh,self._arr[submesh._slice])
    
    def differentiate(self):
        return MidFunction(self._mesh,arr=np.diff(self._arr)/self._mesh._dz)
    
    def integrate(self,flipped=False,output=None):
        # doesn't check that output is MidFunction
        if output is None:
            output=MidFunction(self._mesh)
        np.cumsum(
            (self._arr*self._mesh._dzp)[:-1] if not flipped else np.flipud(-self._arr*self._mesh._dzp)[:-1],
            out=(output.array if not flipped else np.flipud(output.array)))
        return output
        
        
class MidFunction(Function):
    def __init__(self,mesh,arr=np.NaN):
        # doesn't check that mesh and arr are compatible
        self._mesh=mesh
        self._z=mesh.zp #p
        if hasattr(arr,'__iter__'):
            self._arr=arr
        else:
            self._arr=np.array([arr]*len(self._z))
        
    def restrict(self,submesh):
        # doesn't check that submesh and mesh are compatible
        return MidFunction(submesh,self._arr[submesh._slicep]) #p
    
    def to_point_function(self,interp='unweighted'):
        if interp=='unweighted':
            arr=np.empty(len(self._arr)+1)
            arr[1:-1]=(self._arr[1:]+self._arr[:1])/2
            arr[[0,-1]]=arr[[1,-2]]
        if interp=='z':
            arr=interp1d(self._z,self._arr,
                fill_value='extrapolate')(self._mesh.z)
        return PointFunction(self._mesh,arr=arr)
    
    def differentiate(self,fill_value=np.NaN):
        pf=PointFunction(self._mesh)
        pf.array[1:-1]=np.diff(self._arr)/self._mesh._dzp[1:-1]
        pf.array[[0,-1]]=fill_value
        return pf
    
    def integrate(self,flipped=False,output=None):
        # doesn't check that output is PointFunction
        if output is None:
            output=PointFunction(self._mesh)
        output.array[0]=0.0
        np.cumsum(
            self._arr*self._mesh._dz if not flipped else np.flipud(-self._arr*self._mesh._dz),
            out=(output.array[1:] if not flipped else np.flipud(output.array)[1:]))
        return output
    
    #def integrate(self,)

def MaterialFunction(mesh,prop,pos='mid'):
    # could make this more efficient by directly interpolating if Point case?

    ptcounts=np.diff([0]+[i for i,ll,lr in mesh.interfaces]+[len(mesh.z)-1])
    arr=[]

    propfunc=(lambda i: prop(mesh._layers[i].material))\
        if callable(prop)\
        else (lambda i: mesh._layers[i][prop])
    for i,ptc in enumerate(ptcounts):
        arr+=[propfunc(i)]*ptc
        
    out=MidFunction(mesh,arr=np.array(arr))
    if pos=="point":
        return out.to_point_function()
    else: return out
    
def RegionFunction(mesh,prop,pos='mid'):
    # could make this more efficient by directly interpolating if Point case?

    ptcounts=np.diff([0]+[i for i,ll,lr in mesh.interfaces]+[len(mesh.z)-1])
    arr=[]

    propfunc=(lambda i: prop(mesh._layers[i].name))\
        if callable(prop)\
        else (lambda i: mesh._layers[i][prop])
    for i,ptc in enumerate(ptcounts):
        arr+=[propfunc(i)]*ptc
        
    out=MidFunction(mesh,arr=np.array(arr))
    if pos=="point":
        return out.to_point_function()
    else: return out