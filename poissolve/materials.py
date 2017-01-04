# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:20:08 2017

@author: sam
"""
import numpy
from poissolve.constants import m0, eps_0, eV

phib=1

_materials={'GaN':{'name': 'Gallium Nitride', 'abbrev': 'GaN',
            'Eg': 3.605*eV, 'Ei':0*eV, 'DEc': 0*eV,
            'ladder': {
                'electron':{'g':2,'mzs':.2*m0,'mxys':.2*m0, 'mdos': .2*m0},
                # I'll regret this, but there is one hole in GaN... for now
                'hole':{'g':2, 'mdos': 1.5*m0 }},
            'eps': 10.6*eps_0,
            'dopants': {
                'Si':{'type':'Donor','E':.015*eV, 'g':2},
                'Mg':{'type':'Acceptor','E':.230*eV,'g':4},},
            },
           'AlN':{'name': 'Aluminum Nitride', 'abbrev': 'AlN',
            'Eg': 6.14*eV, 'Ei':0, 'DEc': 1.835*eV,
            'ladder': {
                'electron':{'g':2,'mzs':.4*m0,'mxys':.4*m0, 'mdos': .4*m0},
                
                # One hole in AlN as well...this is silly
                'hole':{'g':2,'mdos': 7.26*m0},},
            'eps': 8.6*eps_0,
                  
            # No basis in reality...
            'dopants': {
                'Si':{'type':'Donor','E':4*eV, 'g':2},
                'Mg':{'type':'Acceptor','E':4*eV,'g':4},},
            }}

class Material():
    def __init__(self,matname):
        self._params=_materials[matname]
    
    def __getitem__(self,param):
        if isinstance(param,str): param=[param]
        try:
            d=self._params
            for k in param:
                d=d[k]
            return d
        except:
            raise Exception("Multilevel key error: "+str(param))
    
    def get(self,param,default=None):
        try:
            return self.__getitem__(param)
        except:
            return default
            

if __name__=='__main__':
    import pytest, poissolve
    from tests import test_materials as tester
    pytest.main([tester.__file__])