import numpy as np
from pynitride.paramdb import ParamDB, parr

def unit_cell_volume(mat):
    a=mat['lattice.a']
    c=mat['lattice.c']
    return np.sqrt(3)/2*a**2*c

def density(mat, isotope='mostcommon'):
    pmdb=ParamDB()

    crystal=mat['crystal']

    # sum the masses of the most common isotope for each atom in the basis
    if isotope=='mostcommon':
        mass=\
            sum(pmdb[elt+'.isotope.mass'][np.argmax(pmdb[elt+'.isotope.composition'])]\
                for elt in pmdb[crystal+'.conventional.basis.element'])
    elif isotope=='compositional':
        mass= \
            np.sum(sum([np.multiply(pmdb[elt+'.isotope.mass'],pmdb[elt+'.isotope.composition']) \
                for elt in pmdb[crystal+'.conventional.basis.element']]))
    elif isinstance(isotope,dict):
        mass= \
            sum(pmdb[elt+'.'+str(isotope[elt])+'.mass'] \
                for elt in pmdb[crystal+'.conventional.basis.element'])

    else: raise NotImplementedError

    return mass/unit_cell_volume(mat)

if __name__=='__main__':
    print('hi')
    from pynitride import Material, to_unit
    print("Most common isotope")
    print("Density of GaN: {:.3g}".format(density(Material('GaN',conditions=['relaxed'])).to('g/cm**3')))
    print("Compositionally weighted")
    print("Density of GaN: {:.3g}".format(density(Material('GaN',conditions=['relaxed']),'compositional').to('g/cm**3')))
    print("\nWhat we use in MBE?")
    print("Density of GaN: {:.3g}".format(density(Material('GaN',conditions=['relaxed']),isotope={'Nitrogen':14,'Gallium': 71}).to('g/cm**3')))
