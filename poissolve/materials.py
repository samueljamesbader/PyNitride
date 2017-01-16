# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:20:08 2017

@author: sam
"""
import numpy
from poissolve.constants import m0, eps_0, eV, cm
from poissolve.util import MultilevelDict
import re
from os.path import expanduser

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
            'AlGaN':{'name': 'Aluminum Gallium Nitride', 'abbrev': 'AlGaN',
                   'Eg': (.25*6.14+.75*3.605)*eV, 'Ei':0, 'DEc': .25*(1.835)*eV,
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

                   'barrier':{'GenericMetal':1.5*eV} # hell if I know
                   },
           'GaAs': {
               'DEc':0,
               'ladder':{'electron': {'Gamma': {'g': 2, 'mzs': .067*m0, 'mxys': .067*m0, 'mdos': .067*m0, 'DE': 0}},
                         'hole':{}},
               'eps': 1, #FAKE
               'Eg': 1 #FAKE
           },
            'AlGaAs': { # to mach li kuhn paper
                'DEc':.4*eV,
                'ladder':{'electron': {'Gamma': {'g': 2, 'mzs': .1*m0, 'mxys': .1*m0, 'mdos': .1*m0, 'DE': 0}},
                         'hole':{}},
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


def read_1dp_mat(filename=expanduser("~/1DPoisson/1D Poisson Beta 8g Linux Distribution/Input_Files_Examples/materials.txt")):
    with open(filename) as f:
        for line in f:
            #print(line)
            mo=re.match(r"^(\w+)\s+binary\s+\w+",line.strip())
            if mo:
                matname=mo.groups(0)[0]
                _materials[matname]={}
                #print(matname)
                mo=re.match(
                    "\s+".join(["([\d\.eE\+\-]+)"]*11),next(f))
                Eg, DEc, eps, me, g, mhh, mlh, Ed, Ea, Edd, Eaa=[float(x) for x in mo.groups()]
                mo=re.match(
                    "\s+".join(["([\d\.eE\+\-]+)"]*10),next(f))
                _,_,_,_,_,_,_,_,P,_=[float(x) for x in mo.groups()]

                import scipy.constants as const
                _materials[matname]=dict(
                    name=matname,abbrev=matname,Eg=Eg*eV,Ei=0, DEc=DEc*eV,
                    ladder=dict(
                        electron=dict(
                            Gamma=dict(g=2*g,mzs=me*m0,mxys=me*m0,mdos=me*m0,DE=0)),
                        hole=dict(
                            HH=dict(g=2*g,mzs=mhh*m0,mxys=mhh*m0,mdos=mhh*m0,DE=0),
                            LH=dict(g=2*g,mzs=mlh*m0,mxys=mlh*m0,mdos=mlh*m0,DE=0))),
                    eps=eps*eps_0,dopants=dict(
                        Donor=dict(type='Donor',E=Ed*eV, g=2),
                        Acceptor=dict(type='Acceptor',E=Ea*eV, g=4),
                        DeepDonor=dict(type='Donor',E=Edd*eV, g=2),
                        DeepAcceptor=dict(type='Acceptor',E=Eaa*eV, g=4)),
                    barrier=dict(), P=-P/const.elementary_charge/cm**2)



# _materials={'GaN':{'name': 'Gallium Nitride', 'abbrev': 'GaN',
#                    'Eg': 3.605*eV, 'Ei':0*eV, 'DEc': 0*eV,
#                    'ladder': {
#                        'electron':{'Gamma':{'g':2,'mzs':.2*m0,'mxys':.2*m0, 'mdos': .2*m0, 'DE':0}},
#                        'hole':{
#                            # These values just come from NSM archive, don't trust them
#                            'HH':{'g':2, 'mzs': 1.1*m0, 'mxys': 1.6*m0, 'mdos': 1.5*m0, 'DE':0 },
#                            'LH':{'g':2, 'mzs': 1.1*m0, 'mxys': .15*m0, 'mdos': 1.5*m0, 'DE':0 },
#                            'CH':{'g':2, 'mzs': .15*m0, 'mxys': 1.1*m0, 'mdos': 1.5*m0, 'DE':.02 },
#                        }},
#                    'eps': 10.6*eps_0,
#                    'dopants': {
#                        'Si':{'type':'Donor','E':.015*eV, 'g':2},
#                        'Mg':{'type':'Acceptor','E':.230*eV,'g':4},},
#                    'barrier':{'GenericMetal':1*eV}
#                    },






if __name__=='__main__':
    read_1dp_mat()
    #import pytest, poissolve
    #from poissolve.tests import test_materials as tester
    #pytest.main([tester.__file__])