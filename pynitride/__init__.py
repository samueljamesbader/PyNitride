import os.path
import configparser
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
config = configparser.ConfigParser()
config.read(os.path.join(ROOT_DIR,"config.ini"))

from pynitride.paramdb import ParamDB, Material, Value

from pynitride.poissolve.mesh import Mesh, SubMesh, EpiStack, PointFunction, MidFunction, ConstantFunction, \
    MaterialFunction, RegionFunction

from pynitride.omniscient.dataman import DataManager

