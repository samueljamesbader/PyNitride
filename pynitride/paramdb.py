from pynitride import ROOT_DIR
import os.path
import pint
import re
import scipy.constants as const
import numbers
import numpy as np
from omniscient import Brain

# This is ugly
Brain._ureg=pint.UnitRegistry(system='neu')
Brain._ureg.load_definitions(os.path.join(ROOT_DIR,"parameters","_system.txt"))

def parse(val):
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

    :param val: A numerical quantity which is in the internal unit system of PyNitride
    :param unit: The desired units for the result as a string (eg "cm**-2")
    :return: the number representing that quantity in the desired units.
    """
    return val/Brain._ureg(unit).magnitude

class ParamDB(Brain):
    def __getitem__(self,key):
        val=super().__getitem__(key)
        if isinstance(val,str):
            return val
        else:
            return Brain._ureg.Quantity(val).to_base_units().magnitude
    #def quantity(self,*args):
    #    # If it's a single array, and not made of numeric or Pint quantity elements, just return as is
    #    if len(args)==1 and hasattr(args[0],'__getitem__') and not isinstance(args[0],str):
    #        if args[0]==[]: return np.array([])
    #        elif not (isinstance(args[0][0],numbers.Number) or hasattr(args[0][0],"units")):
    #            return args[0]

    #    if self._units=='neu':
    #        if len(args)==1 and isinstance(args[0],str):
    #            if "," in args[0]:
    #                return [self.quantity(s) for s in args[0].split(',')]
    #        elif len(args)==1 and hasattr(args[0],'__getitem__'):
    #            return np.array([self.quantity(a) for a in args[0]])
    #        return ParamDB._ureg.Quantity(*args).to_base_units().magnitude

k,hbar,pi,m_e,cm,nm,eV,K,q =[parse(x) for x in "k,hbar,pi,m_e,cm,nm,eV,K,e".split(',')]
pmdb=ParamDB(os.path.join(ROOT_DIR,"parameters"))
pmdb.read("VM2003.txt")
pmdb.read("BFV2001.txt")
pmdb.read("fake.txt")
#print("WHAT")
#print(pmdb['GaN.dielectric.eps'])
#print(pmdb['GaN.dopant'])
#print("ME")
#exit()
