import numpy as np
from pynitride.mesh import Function
from functools import partial
from pynitride.machine import glob_store_attributes

def polar2cart(rho,theta):
    """ Converts a polar rho,theta coordinate to a Cartesian x,y coordinate.

    Arguments rho and theta can be scalars/arrays/anything that numpy accepts.
    Normal broadcasting rules apply.

    Args:
        rho: radial coordinate
        theta: angular coordinate in radians

    Returns:
        tuple of (x, y)
    """
    return rho*np.cos(theta),rho*np.sin(theta)

def cart2polar(x,y):
    """ Converts a Cartesian x,y coordinate to a polar rho,theta coordinate.

    Arguments x and y can be scalars/arrays/anything that numpy accepts.
    Normal broadcasting rules apply.

    Args:
        x: x coordinate
        y: y coordinate

    Returns:
        tuple of (rho, theta)
    """
    return np.sqrt(x**2+y**2),np.arctan2(y,x)



def double_mat(arr, dtype='complex'):
    m=arr.mesh
    if len(arr.shape)>2:
        n=arr.shape[0]
        out=Function(m,value=np.zeros((2*n,2*n,arr.shape[2]),dtype=dtype),dtype=dtype,pos=arr.pos)
        out[:n,:n,:]=arr
        out[n:,n:,:]=arr
        return out
    else:
        # this branch actually doesn't work but is never used, should delete
        n=arr.shape[0]
        out=Function(m,value=np.zeros((2*n,2*n),dtype=dtype),dtype=dtype,pos=arr.pos)
        out[:n,:n]=arr
        out[n:,n:]=arr
        return out

def chunks(lst, n): 
    """Yield successive n-sized chunks from lst."""
    return [list(lst[i:i + n]) for i in range(0,len(lst),n)]

def dephase(vec):
    i=np.argmax(np.abs(vec),axis=-1)
    phase=vec.T[i]/np.abs(vec.T[i])
    return (vec.T/phase).T

def round_near(arr,ival):
    return np.round(np.asarray(arr)/ival,0)*ival

@glob_store_attributes("_cache")
class Memoizer:
    def __init__(self):
        self._cache={}
    def mfunc(self,func,name,*args,**kwargs):
        class hashabledict(dict):
            def __key(self):
                return tuple((k,self[k]) for k in sorted(self))
            def __hash__(self):
                return hash(self.__key())
            def __eq__(self, other):
                return self.__key() == other.__key()
        key=(args,hashabledict(kwargs))
        if key not in self._cache[name]:
            self._cache[name][key]=func(*args,**kwargs)
        else:
            print("Got {} from cache".format(name))
        return self._cache[name][key]
    def memoize(self,func,name):
        if name not in self._cache: self._cache[name]={}
        return partial(self.mfunc,func,name)
memoizer=Memoizer()
