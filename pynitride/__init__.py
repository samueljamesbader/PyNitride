import os.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

from pynitride.paramdb import ParamDB, Material, Value, parr
to_unit=Value.to_unit
parse=Value.parse
del Value

convenient_constants=["hbar","c","m_e","angstrom","nm","um","mm","cm","mV","V","kV","MV","meV","eV","keV","MeV","epsilon_0"]
for const in convenient_constants:
    globals()[const]=parse(const)
q=parse('e')
T=300
kT=parse('k')*T
del convenient_constants

from pynitride.poissolve.mesh.structure import Mesh, SubMesh, EpiStack
from pynitride.poissolve.mesh.functions import MaterialFunction, MidFunction, PointFunction, ConstantFunction, RegionFunction
