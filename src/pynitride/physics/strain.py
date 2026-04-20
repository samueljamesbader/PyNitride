import numpy as np
from pynitride import sublog

class Pseudomorphic():

    def __init__(self,mesh,straincond=None):
        self._mesh=mesh
        self._straincond=straincond
        self._mesh.ensure_function_exists('exx',0,pos='mid')
        self._mesh.ensure_function_exists('eyy',0,pos='mid')
        self._mesh.ensure_function_exists('ezz',0,pos='mid')
        self._mesh.ensure_function_exists('exy',0,pos='mid')
        self._mesh.ensure_function_exists('exz',0,pos='mid')
        self._mesh.ensure_function_exists('eyz',0,pos='mid')


    def initialize(self):
        return self

    def solve(self):
        straincond=self._straincond
        if straincond is None:
            pos=-1 if (self._mesh.ztrans == -1 ) else 0
            straincond=self._mesh._matblocks[pos].matsys.bulk_lattice_condition(self._mesh._matblocks[pos].mesh)
        for matblock in self._mesh._matblocks:
            matblock.matsys.strain_to(matblock.mesh,straincond=straincond)
            matblock.update(self._mesh,'strain')

