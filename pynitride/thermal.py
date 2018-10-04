import numpy as np
from pynitride.paramdb import K
from pynitride.mesh import MidFunction

class ConstantT():

    def __init__(self,mesh,T=300.*K):
        self._mesh=mesh
        self._T=T
        mesh.add_attr('T',self.solve)

    def initialize(self):
        pass

    def solve(self):
        self._mesh['T']=MidFunction(self._mesh,value=self._T)
        #for mb in self._mesh._matblocks:
        #    mb.update('temperature',self._mesh)
