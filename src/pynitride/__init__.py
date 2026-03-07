import os.path
from configparser import ConfigParser
from multiprocessing import get_all_start_methods

# Needed for reading configuration
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))

def _make_default_config(path):
    cp = ConfigParser()
    parallel = 'fork' in get_all_start_methods()
    cp.add_section("parallelism")
    cp.set("parallelism", "globalthreads", "cpu_count")
    cp.set("parallelism", "globalprocesses", "cpu_count" if parallel else "1")
    cp.set("parallelism", "cextthread", "1" if parallel else "DEFAULT")
    cp.add_section("logging")
    cp.set("logging", "level", "info")
    with open(path, 'w') as f:
        cp.write(f)

_config_path = os.path.join(ROOT_DIR, "config.ini")
if not os.path.exists(_config_path):
    _make_default_config(_config_path)

config = ConfigParser()
config.read(_config_path)
""" The directory of the PyNitride project"""

# Logging is needed everywhere and doesn't require anything (eg numpy)
# which would preempt the configuration of the parallelism
from pynitride.core.logging import log, sublog

# With that in-place we can set up parallelism
# before any other scripts which would preempt its configuration
# eg by importing numpy
import pynitride.core.machine

# Common core components
from pynitride.core.paramdb import pmdb, parse, to_unit, kb, hbar, pi, m_e, cm, nm, eV, meV, K, q
from pynitride.core.mesh import Mesh, SubMesh, UniformLayer, GradedLayer, MaterialBlock
from pynitride.core.mesh import NodFunction, MidFunction, MaterialFunction, Function
from pynitride.core.reciprocal_mesh import RMesh1D, RMesh2D_Polar

# Common physics components
from pynitride.physics.carriers import Semiclassical, Schrodinger, MultibandKP
from pynitride.physics.thermal import ConstantT
from pynitride.physics.strain import Pseudomorphic
from pynitride.physics.solvers import PoissonSolver, Linear_Fermi, Equilibrium, SelfConsistentLoop

# Overall flow
from pynitride.core.sim import Simulation
