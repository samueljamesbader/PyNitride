import os.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

import pynitride.paramdb
#from pynitride.paramdb import ParamDB, Material, to_unit, parse, convenient_constants
#for const in convenient_constants: globals()[const]=getattr(pynitride.paramdb,const)
from pynitride.poissolve.mesh.structure import Mesh, SubMesh, EpiStack
from pynitride.poissolve.mesh.functions import MaterialFunction, MidFunction, PointFunction, ConstantFunction, RegionFunction
