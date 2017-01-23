import numpy as np
from pynitride.paramdb import ParamDB

def density(mat):
    pdb=ParamDB.get_global()
    pdb.read_file('chemistry.txt')

    a=mat['lattice','a']
    c=mat['lattice','c']
    V_unitcell=np.sqrt(3)/2*a**2*c

    mass_unitcell= sum([pdb['element'][element]['mass']*num for element, num in mat['crystal','basis atoms'].items()])

    return mass_unitcell/V_unitcell
