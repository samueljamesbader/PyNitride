import numpy as np

class Pseudomorphic():

    def __init__(self,mesh,straincond=None):
        self._mesh=mesh
        self._straincond=straincond
        mesh.add_attr('exx',self.solve)
        mesh.add_attr('eyy',self.solve)
        mesh.add_attr('ezz',self.solve)

    def initialize(self):
        return self

    def solve(self):
        straincond=self._straincond
        if straincond is None:
            pos=-1 if (self._mesh.ztrans == -1 ) else 0
            straincond=self._mesh._matblocks[pos].matsys.bulk_lattice_condition(self._mesh._matblocks[pos].mesh)
        for matblock in self._mesh._matblocks:
            matblock.matsys.strain_to(matblock.mesh,straincond=straincond)

        #for mb in self._mesh._matblocks:
        #    mb.update('strain',self._mesh)

