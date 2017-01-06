# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 23:13:27 2017

@author: sam
"""

import numpy as np
import re
from poissolve.constants import kT,hbar,q
from poissolve.mesh_functions import PointFunction, MaterialFunction
from poissolve.maths.fermi_dirac_integral import fd12, fd12p
from poissolve.materials import _materials

class FermiDirac3D():
    def __init__(self,mesh,compute_dopants='GaN'):
        self._mesh=mesh
        #mesh['n']=PointFunction(mesh)
        #mesh['p']=PointFunction(mesh)

        # We'll have to confirm these formulae (factors of 2?) later... also one hole in GaN?
        self._Nc=MaterialFunction(mesh,pos='point',prop=lambda mat:
            mat['ladder','electron','g']*(mat['ladder','electron','mdos']*kT/(2*np.pi*hbar**2))**(3/2))
        self._Nv=MaterialFunction(mesh,pos='point',prop=lambda mat:
            mat['ladder']['hole']['g']*(mat['ladder']['hole']['mdos']*kT/(2*np.pi*hbar**2))**(3/2))

        #self._nderiv=mesh.add_function('nderiv',PointFunction(mesh))
        #self._pderiv=mesh.add_function('pderiv',PointFunction(mesh))
        #self._Namderiv=PointFunction(mesh)
        #self._Ndpderiv=PointFunction(mesh)
        #self._rhoderiv=mesh.add_function('rhoderiv',PointFunction(mesh))


        self._identifydopants()

    def _identifydopants(self):

        m=self._mesh
        self._dopants={'Donor':{},'Acceptor':{}}
        for k in m:
            mo=re.match("(.*)ActiveConc",k)
            if mo and mo.group(1) not in self._dopants:
                d=mo.group(1)
                types=set(l.material.get(['dopants',d,'type'],None) for l in m._layers)
                assert len(types)<2, "Can't have one dopant be acceptor in one material and donor in another.  You'll have " \
                                   "to use two separate dopant names.  Sorry. "
                if len(types)==1:
                    self._dopants[list(types)[0]][d]={'conc':m[k]}
                else:
                    print("No materials include {} as a dopant.".format(k))
        for doptype in self._dopants.keys():
            for d,v in self._dopants[doptype].items():
                v['E']=MaterialFunction(m,lambda mat: mat.get(['dopants',d,'E'],np.NaN),pos='point')
                v['g']=MaterialFunction(m,lambda mat: mat.get(['dopants',d,'g'],np.NaN),pos='point')

        m['Nd']=np.sum(d['conc'] for d in self._dopants['Donor'].values())
        m['Na']=np.sum(d['conc'] for d in self._dopants['Acceptor'].values())

    def solve(self,activation=1):
        m=self._mesh
        EF=m['EF']
        Ec=m['Ec']
        Ev=m['Ev']

        m['Ndp']=np.sum( d['conc']*(1/(1+d['g']*np.exp((EF-Ec+d["E"])/kT)))
            for d in self._dopants['Donor'].values())
        m['Nam']=np.sum( d['conc']*(1/(1+d['g']*np.exp((Ev+d["E"]-EF)/kT)))
            for d in self._dopants['Acceptor'].values())
        m['Ndpderiv']=np.sum( d['conc']*(d['g']/kT)*np.exp((EF-Ec+d["E"])/kT)/(1+d['g']*np.exp((EF-Ec+d["E"])/kT))**2
            for d in self._dopants['Donor'].values())
        m['Namderiv']=np.sum( d['conc']*(-d['g']/kT)*np.exp((Ev+d["E"]-EF)/kT)/(1+d['g']*np.exp((Ev+d["E"]-EF)/kT))**2
            for d in self._dopants['Acceptor'].values())

        m['n']=self._Nc*fd12((EF-Ec)/kT)
        m['p']=self._Nv*fd12((Ev-EF)/kT)
        m['nderiv']=-(self._Nc/kT)*fd12p((EF-Ec)/kT)
        m['pderiv']= (self._Nv/kT)*fd12p((Ev-EF)/kT)

        temp_rho=m['rho_pol']+q*(m['p']+m['Ndp']-m['n']-m['Nam'])

        m['rho']=activation*temp_rho
        m['rhoderiv']= activation*q*(m['pderiv']+m['Ndpderiv']-m['nderiv']-m['Namderiv'])

