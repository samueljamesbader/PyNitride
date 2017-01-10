# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 23:13:27 2017

@author: sam
"""

import re

import numpy as np
from poissolve.maths.fermidiracintegral import fd12, fd12p

from poissolve.constants import kT,hbar,q
from poissolve.mesh.functions import MaterialFunction, PointFunction


class FermiDirac3D():
    def __init__(self,mesh):
        self._mesh=mesh
        #mesh['n']=PointFunction(mesh)
        #mesh['p']=PointFunction(mesh)

        # We'll have to confirm these formulae (factors of 2?) later... also one hole in GaN?
        self._Nc=MaterialFunction(mesh,pos='point',prop=lambda mat:
            [mat['ladder','electron',b,'g']*(mat['ladder','electron',b,'mdos']*kT/(2*np.pi*hbar**2))**(3/2)
                for b in mat['ladder','electron']])
        self._Nv=MaterialFunction(mesh,pos='point',prop=lambda mat:
            [mat['ladder','hole',b,'g']*(mat['ladder','hole',b,'mdos']*kT/(2*np.pi*hbar**2))**(3/2)
                for b in mat['ladder','hole']])
        mesh['cDE']=self._cDE=MaterialFunction(mesh,pos='point',prop=lambda mat:
            [mat['ladder','electron',b,'DE'] for b in mat['ladder','electron']])
        mesh['vDE']=self._vDE=MaterialFunction(mesh,pos='point',prop=lambda mat:
            [mat['ladder','hole',b,'DE'] for b in mat['ladder','hole']])

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
                types=set(t for t in (l.material.get(['dopants',d,'type'],None) for l in m._layers) if t is not None)
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

    def solve(self,activation=1,quantized_bands=[]):
        m=self._mesh
        EF=m['EF']
        Ec=m['Ec']
        Ev=m['Ev']
        Ec_eff=m['Ec_eff'] if 'electron' in quantized_bands else m['Ec']+self._cDE
        Ev_eff=m['Ev_eff'] if 'hole' in quantized_bands else m['Ev']-self._vDE

        m['Ndp']=PointFunction(m,np.nan_to_num(np.sum( d['conc']*(1/(1+d['g']*np.exp((EF-Ec+d["E"])/kT)))
            for d in self._dopants['Donor'].values())))
        m['Nam']=PointFunction(m,np.nan_to_num(np.sum( d['conc']*(1/(1+d['g']*np.exp((Ev+d["E"]-EF)/kT)))
            for d in self._dopants['Acceptor'].values())))
        m['Ndpderiv']=PointFunction(m,np.nan_to_num(np.sum( d['conc']*(d['g']/kT)*np.exp((EF-Ec+d["E"])/kT)/(1+d['g']*np.exp((EF-Ec+d["E"])/kT))**2
            for d in self._dopants['Donor'].values())))
        m['Namderiv']=PointFunction(m,np.nan_to_num(np.sum( d['conc']*(-d['g']/kT)*np.exp((Ev+d["E"]-EF)/kT)/(1+d['g']*np.exp((Ev+d["E"]-EF)/kT))**2
            for d in self._dopants['Acceptor'].values())))

        m['n']=np.sum(self._Nc*fd12((EF-Ec_eff)/kT),axis=0) + (m['n_quantum'] if 'n_quantum' in m else 0)
        m['p']=np.sum(self._Nv*fd12((Ev_eff-EF)/kT),axis=0) + (m['p_quantum'] if 'p_quantum' in m else 0)
        m['nderiv']=np.sum(-(self._Nc/kT)*fd12p((EF-Ec_eff-self._cDE)/kT),axis=0) + (m['nderiv_quantum'] if 'nderiv_quantum' in m else 0)
        m['pderiv']=np.sum((self._Nv/kT)*fd12p((Ev_eff+self._vDE-EF)/kT),axis=0) + (m['pderiv_quantum'] if 'pderiv_quantum' in m else 0)

        temp_rho=m['rho_pol']+q*(m['p']+m['Ndp']-m['n']-m['Nam'])

        m['rho']=activation*temp_rho
        m['rhoderiv']= activation*q*(m['pderiv']+m['Ndpderiv']-m['nderiv']-m['Namderiv'])

