# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:20:08 2017

@author: sam
"""
import numpy
from pynitride.paramdb import m_e, eps_0, eV, cm
from pynitride.paramdb import MultilevelDict
import re
from os.path import expanduser
import numpy as np

def read_1dp_mat(filename=expanduser("/usr/local/bin/materials.txt")):
    with open(filename) as f:
        for line in f:
            #print(line)
            mo=re.match(r"^(\w+)\s+binary\s+\w+",line.strip())
            if mo:
                matname=mo.groups(0)[0]
                _materials[matname]={}

                tmp={}
                next(f) # skip mystery zeros line in materials file
                for line in f:
                    mo=re.match(r"(\w+)=([\deE\+\-\.]+)",line)
                    if mo is None: break
                    try:
                        tmp[mo.groups()[0]]=float(mo.groups()[1])
                    except:
                        tmp[mo.groups()[0]]=np.NaN

                import scipy.constants as const
                _materials[matname]=dict(
                    name=matname,abbrev=matname,Eg=tmp['eg']*eV,Ei=0, DEc=tmp['dec']*eV,
                    ladder=dict(
                        electron=dict(
                            Gamma=dict(g=2*tmp['val'],mzs=tmp['me']*m_e,mxys=tmp['me']*m_e,mdos=tmp['me']*m_e,DE=0)),
                        hole=dict(
                            HH=dict(g=2,mzs=tmp['mh']*m_e,mxys=tmp['mh']*m_e,mdos=tmp['mh']*m_e,DE=0),
                            LH=dict(g=2,mzs=tmp['mlh']*m_e,mxys=tmp['mlh']*m_e,mdos=tmp['mlh']*m_e,DE=0),
                            SO=dict(g=2,mzs=tmp['mhso']*m_e,mxys=tmp['mhso']*m_e,mdos=tmp['mhso']*m_e,DE=0))),
                    eps=tmp['er']*eps_0,dopants=dict(
                        Donor=dict(type='Donor',E=tmp['ed']*eV, g=2),
                        Acceptor=dict(type='Acceptor',E=tmp['ea']*eV, g=4),
                        DeepDonor=dict(type='Donor',E=tmp['edd']*eV, g=2),
                        DeepAcceptor=dict(type='Acceptor',E=tmp['eda']*eV, g=4)),
                    barrier=dict(), P=-tmp['pol']/const.elementary_charge/cm**2)


if __name__=='__main__':
    read_1dp_mat()
    #import pytest, poissolve
    #from poissolve.tests import test_materials as tester
    #pytest.main([tester.__file__])