__version__ = "0.2.0rc4"

import os.path
from dotenv import load_dotenv
load_dotenv()

from types import MappingProxyType
from typing import cast

# Utility for an immutable empty dict, to avoid the pitfalls of mutable default arguments
_EMPTYD = cast(dict,MappingProxyType({}))

# Logging is needed everywhere and doesn't require anything (eg numpy)
# which would preempt the configuration of the parallelism
from pynitride.core.logging_ import log, sublog

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
