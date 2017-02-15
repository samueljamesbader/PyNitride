import xrayutilities as xu
from pynitride import ParamDB, Material

def _make_materials():
    pmdb=ParamDB(units='Pint')
    GaN=Material("GaN",pmdb=pmdb)
    AlN=Material("AlN",pmdb=pmdb)

    Ga=xu.materials.elements.Ga
    N=xu.materials.elements.N
    Al=xu.materials.elements.Al
    Si = xu.materials.elements.Si
    C = xu.materials.elements.C

    a_GaN=GaN['lattice.a'].to('angstrom').magnitude
    c_GaN=GaN['lattice.c'].to('angstrom').magnitude
    u_GaN=GaN['lattice.u']
    a_AlN=AlN['lattice.a'].to('angstrom').magnitude
    c_AlN=AlN['lattice.c'].to('angstrom').magnitude
    u_AlN=AlN['lattice.u']

    GaN_WZ = xu.materials.Crystal(
        "GaN(WZ)", xu.materials.SGLattice(186,a_GaN,c_GaN,
            atoms=[Ga, N], pos=[('2b', 0),
                                ('2b', u_GaN)]),
        xu.materials.HexagonalElasticTensor(GaN['stiffness.C11'].to('pascal').magnitude,
                                            GaN['stiffness.C12'].to('pascal').magnitude,
                                            GaN['stiffness.C13'].to('pascal').magnitude,
                                            GaN['stiffness.C33'].to('pascal').magnitude,
                                            GaN['stiffness.C44'].to('pascal').magnitude))
    AlN_WZ = xu.materials.Crystal(
        "AlN(WZ)", xu.materials.SGLattice(186,a_AlN,c_AlN,
                                          atoms=[Al, N], pos=[('2b', 0),
                                                              ('2b', u_AlN)]),
        xu.materials.HexagonalElasticTensor(AlN['stiffness.C11'].to('pascal').magnitude,
                                            AlN['stiffness.C12'].to('pascal').magnitude,
                                            AlN['stiffness.C13'].to('pascal').magnitude,
                                            AlN['stiffness.C33'].to('pascal').magnitude,
                                            AlN['stiffness.C44'].to('pascal').magnitude))

    SiC_6H = xu.materials.Crystal(
        "SiC(6H)",xu.materials.SGLattice(186, 3.0810, 15.1248,
                                         atoms=[Si,C,Si,C,Si,C], pos=[('2b',0.20796),
                                                                      ('2b',0.33323),
                                                                      ('2b',0.54151),
                                                                      ('2b',0.66661),
                                                                      ('2a',0.00000),
                                                                      ('2a',0.37473)]),
        xu.materials.HexagonalElasticTensor(501e9,111e9,52e9,553e9,163e9))
    return GaN_WZ, AlN_WZ, SiC_6H
GaN_WZ,AlN_WZ,SiC_6H=_make_materials()

def PseudomorphicStack0001(name,*args):
    return xu.simpack.PseudomorphicStack001(name,*args)
Layer=xu.simpack.Layer
