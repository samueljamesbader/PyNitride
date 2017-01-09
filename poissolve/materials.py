# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:20:08 2017

@author: sam
"""
import numpy
from poissolve.constants import m0, eps_0, eV
from poissolve.util import MultilevelDict

phib=1

_materials={'GaN':{'name': 'Gallium Nitride', 'abbrev': 'GaN',
            'Eg': 3.605*eV, 'Ei':0*eV, 'DEc': 0*eV,
            'ladder': {
                'electron':{'Gamma':{'g':2,'mzs':.2*m0,'mxys':.2*m0, 'mdos': .2*m0, 'DE':0}},
                'hole':{
                    # These values just come from NSM archive, don't trust them
                    'HH':{'g':2, 'mzs': 1.1*m0, 'mxys': 1.6*m0, 'mdos': 1.5*m0, 'DE':0 },
                    'LH':{'g':2, 'mzs': 1.1*m0, 'mxys': .15*m0, 'mdos': 1.5*m0, 'DE':0 },
                    'CH':{'g':2, 'mzs': .15*m0, 'mxys': 1.1*m0, 'mdos': 1.5*m0, 'DE':.02 },
                }},
            'eps': 10.6*eps_0,
            'dopants': {
                'Si':{'type':'Donor','E':.015*eV, 'g':2},
                'Mg':{'type':'Acceptor','E':.230*eV,'g':4},},
            'barrier':{'GenericMetal':1*eV}
            },
           'AlN':{'name': 'Aluminum Nitride', 'abbrev': 'AlN',
            'Eg': 6.14*eV, 'Ei':0, 'DEc': 1.835*eV,
            'ladder': {
                'electron':{'Gamma':{'g':2,'mzs':.4*m0,'mxys':.4*m0, 'mdos': .4*m0, 'DE': 0}},
                
                # One hole in AlN as well...this is silly
                'hole':{
                    # These values just come from NSM archive, don't trust them
                    'HH':{'g':2, 'mzs': 3.5*m0, 'mxys': 10.4*m0, 'mdos': 7.26*m0, 'DE':0 },
                    'LH':{'g':2, 'mzs': 3.5*m0, 'mxys': 0.24*m0, 'mdos': 0.58*m0, 'DE':0 },
                    'CH':{'g':2, 'mzs': .25*m0, 'mxys': 3.81*m0, 'mdos': 1.54*m0, 'DE':.019*eV },
             }},
            'eps': 8.6*eps_0,
                  
            # No basis in reality...
            'dopants': {
                'DeepDonor':{'type':'Donor','E':4*eV, 'g':2},
                'DeepAcceptor':{'type':'Acceptor','E':4*eV,'g':4},},

            'barrier':{'GenericMetal':3*eV} # hell if I know
            },
           'GaAs': {
               'DEc':0,
               'ladder':{'electron': {'Gamma': {'g': 2, 'mzs': .067*m0, 'mxys': .067*m0, 'DE': 0}}},
               'eps': 1, #FAKE
               'Eg': 1 #FAKE
           },
            'AlGaAs': { # to mach li kuhn paper
                'DEc':.4*eV,
                'ladder':{'electron': {'Gamma': {'g': 2, 'mzs': .1*m0, 'mxys': .1*m0, 'DE': 0}}},
                'barrier':{'GenericMetal':.0*eV},
                'eps': 1, #FAKE
                'Eg': 1 #FAKE
            }
        }

# http://stackoverflow.com/a/25176504/2081118
class Material(MultilevelDict):
    def __init__(self,matname):
        super().__init__(_materials[matname])
        self._matname=matname
    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return self._matname==other._matname
        return NotImplemented
    def __neq__(self,other):
        if isinstance(other,self.__class__):
            return self._matname!=other._matname
        return NotImplemented
    def __hash__(self):
        return hash(self._matname)


# class Material():
#     def __init__(self,matname):
#         self._params=_materials[matname]
#
#     def __getattr__(self, attr):
#         if attr in ["get","__getitem__"]:
#             return getattr(self._params,attr)

if __name__=='__main__':
    import pytest, poissolve
    from poissolve.tests import test_materials as tester
    pytest.main([tester.__file__])