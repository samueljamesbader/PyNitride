import os.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

from pynitride.paramdb import ParamDB, Material, Value, parr
to_unit=Value.to_unit
parse=Value.parse
del Value

variables=["hbar","c","m_e","angstrom","nm","um","mm","cm","mV","V","kV","MV","meV","eV","keV","MeV","epsilon_0","e","k"]
ParamDB().make_accessible(globals(),variables)
q=e
T=300
kT=k*T
del variables

from pynitride.poissolve.mesh.structure import Mesh, SubMesh, EpiStack
from pynitride.poissolve.mesh.functions import MaterialFunction, MidFunction, PointFunction, ConstantFunction, RegionFunction
