import numpy as np

class Pseudomorphic():

    def __init__(self,mesh):
        self._mesh=mesh

    def solve(self,straincond=None):
        if straincond is None:
            pos=-1 if (self._mesh.ztrans == -1 ) else 0
            straincond=self._mesh._matblocks[pos].matsys.bulk_lattice_condition(self._mesh._matblocks[pos].mesh)
        for matblock in self._mesh._matblocks:
            matblock.matsys.strain_to(matblock.mesh,straincond=straincond)

