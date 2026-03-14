import os.path
import pint
from pynitride.core.omniscient import Brain
from importlib import resources

# Configures Pint to use "nanoelectronic units"
# The base unit of length is  1 nm
# The base unit of mass is 1 eV fs**2 / nm**2
# The base unit of time is 1 fs
# The base unit of current is 1 e/fs
# In this system, e=1, eV=1, nm=1, fs=1
Brain._ureg=pint.UnitRegistry(system='neu')
Brain._ureg.load_definitions(
    """
    e_per_fs  = e / fs
    eV_fs2_per_nm2 = eV fs**2 / nm**2

    @system neu using international
        nanometer
        eV_fs2_per_nm2
        femtosecond
        e_per_fs
    @end
    """.splitlines())

def parse(val):
    """ Reads the value in as a string and returns a number in nanoelectronic units"""
    v=Brain._ureg(val)
    return v.to_base_units().magnitude

def to_unit(val,unit):
    r""" Convert a number from PyNitride's internal units to any other units.

    Any number used in PyNitride or pulled from the parameter database is assumed to be in "nanoelectronic units", and
    this function provides conversion to other units for outputing readable results to a user.
    Note that this function does not and could not possibly ensure that your conversion is dimensionally valid, since
    the val input is just a number.  The user is responsible for knowing what val is (eg a distance, or an energy etc)
    and requesting sensible output dimensions.  The output dimensions are interpreted by Pint and a full list of allowed
    values (including units such as ``meter``, prefixed units such as ``meV``, more complex units like ``cm**-2`` and
    constants such as ``hbar``) is provided in the
    `Pint docs <https://github.com/hgrecco/pint/blob/master/pint/default_en.txt>`_.

    Args:
        val: A numerical quantity which is in the internal unit system of PyNitride
        unit: The desired units for the result as a string (eg "cm**-2")
    Returns:
        the number representing that quantity in the desired units.
    """
    return val/Brain._ureg(unit).to_base_units().magnitude

class ParamDB(Brain):
    def __getitem__(self,key):
        val=super().__getitem__(key)
        if isinstance(val,str):
            return val
        else:
            return Brain._ureg.Quantity(val).to_base_units().magnitude

# Get some constants
kb, hbar, pi, m_e, cm, nm, eV, meV, K, q =[parse(x) for x in\
        "k,hbar,pi,m_e,cm,nm,eV,meV,K,e".split(',')]

# Read in some parameter files
pmdb=ParamDB(None)
with resources.as_file(resources.files('pynitride.parameters')) as param_path:
    pmdb.read(param_path/"VM2003.txt")
    pmdb.read(param_path/"fake.txt")
    pmdb.read(param_path/"bader_recommended.txt")
