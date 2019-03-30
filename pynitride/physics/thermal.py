import numpy as np
from pynitride import K
from pynitride import MidFunction

class ConstantT():

    def __init__(self,mesh,T=300.*K):
        self._mesh=mesh
        self._T=T
        self._mesh.ensure_function_exists('T',0,pos='mid')

    def initialize(self):
        pass

    def solve(self):
        self._mesh['T']=MidFunction(self._mesh,value=self._T)
        #for mb in self._mesh._matblocks:
        #    mb.update('temperature',self._mesh)
