import numpy as np
from pynitride.paramdb import ParamDB

def unit_cell_volume(mat):
    a=mat['lattice.a']
    c=mat['lattice.c']
    V_unitcell=np.sqrt(3)/2*a**2*c

def density(mat, isotope='mostcommon'):
    pdb=ParamDB()

    crystal=mat['crystal']

    if isotope=='mostcommon':
        # sum the masses of the most common isotope for each atom in the basis
        mass=\
            sum(pdb[elt+'.isotope.mass'][np.argmax(pdb[elt+'.isotope.composition'])]\
                for elt in pdb[crystal+'.conventional.basis.:.element'])
    else: raise NotImplementedError

    return mass/unit_cell_volume(mat)

if __name__=='__main__':
    from pynitride import Material, to_unit
    print("Density of GaN: {:.2g}".format(to_unit(density(Material('GaN',conditions=['relaxed'])),'g/cm**2')))
