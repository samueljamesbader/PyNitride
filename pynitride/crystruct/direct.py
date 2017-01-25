import numpy as np
from pynitride.paramdb import ParamDB

def unit_cell_volume(mat):
    a=mat['lattice','a']
    c=mat['lattice','c']
    V_unitcell=np.sqrt(3)/2*a**2*c

def density(mat):
    pdb=ParamDB()
    mass_unitcell= sum([pdb['element'][element]['mass']*num for element, num in mat['crystal','basis atoms'].items()])
    return mass_unitcell/unit_cell_volume(mat)

