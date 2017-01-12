# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 23:13:27 2017

@author: sam
"""

import re

import numpy as np
from poissolve.maths.fermidiracintegral import fd12, fd12p

from poissolve.constants import kT,hbar,q
from poissolve.mesh.functions import MaterialFunction, PointFunction, ConstantFunction


class FermiDirac3D():
    def __init__(self,mesh):
        self._mesh=mesh

        self._Nc,self._Nv=self.effective_dos_3d(mesh)
        self._cDE,self._vDE=self.band_edge_shifts()
        self._dopants=self.identifydopants(mesh)


    @staticmethod
    def identifydopants(mesh):
        dopants={'Donor':{},'Acceptor':{}}
        for k in mesh:
            mo=re.match("(.*)ActiveConc",k)
            if mo and mo.group(1) not in dopants:
                d=mo.group(1)
                types=set(t for t in (l.material.get(['dopants',d,'type'],None) for l in mesh._layers) if t is not None)
                assert len(types)<2, "Can't have one dopant be acceptor in one material and donor in another.  You'll have " \
                                   "to use two separate dopant names.  Sorry. "
                if len(types)==1:
                    dopants[list(types)[0]][d]={'conc':mesh[k]}
                else:
                    print("No materials include {} as a dopant.".format(k))
        for doptype in dopants.keys():
            for d,v in dopants[doptype].items():
                v['E']=MaterialFunction(mesh,lambda mat: mat.get(['dopants',d,'E'],np.NaN),pos='point')
                v['g']=MaterialFunction(mesh,lambda mat: mat.get(['dopants',d,'g'],np.NaN),pos='point')
        mesh['Nd']=np.sum(d['conc'] for d in dopants['Donor'].values())\
            if len(dopants['Donor']) else ConstantFunction(mesh,0)
        mesh['Na']=np.sum(d['conc'] for d in dopants['Acceptor'].values()) \
            if len(dopants['Acceptor']) else ConstantFunction(mesh,0)
        return dopants

    @staticmethod
    def effective_dos_3d(mesh):

        # We'll have to confirm these formulae (factors of 2?) later...
        Nc=MaterialFunction(mesh,pos='point',prop=lambda mat:
        [mat['ladder','electron',b,'g']*(mat['ladder','electron',b,'mdos']*kT/(2*np.pi*hbar**2))**(3/2)
         for b in mat['ladder','electron']])
        Nv=MaterialFunction(mesh,pos='point',prop=lambda mat:
        [mat['ladder','hole',b,'g']*(mat['ladder','hole',b,'mdos']*kT/(2*np.pi*hbar**2))**(3/2)
         for b in mat['ladder','hole']])
        return Nc,Nv

    @staticmethod
    def band_edge_shifts(mesh):
        cDE=MaterialFunction(mesh,pos='point',prop=lambda mat:
            [mat['ladder','electron',b,'DE'] for b in mat['ladder','electron']])
        vDE=MaterialFunction(mesh,pos='point',prop=lambda mat:
            [mat['ladder','hole',b,'DE'] for b in mat['ladder','hole']])
        return cDE,vDE


    @staticmethod
    def carrier_density(EF,Ec,Ev,Nc,Nv,conduction_band_shifts=0,valence_band_shifts=0,compute_derivs=True):

        # Can save a copy and loop by checking if shifts are zero
        Ec_eff=Ec+conduction_band_shifts
        Ev_eff=Ev-valence_band_shifts

        n=np.sum(Nc*fd12((EF-Ec)/kT),axis=0)
        p=np.sum(Nv*fd12((Ev-EF)/kT),axis=0)

        if compute_derivs:
            nderiv=np.sum(-(Nc/kT)*fd12p((EF-Ec_eff)/kT),axis=0)
            pderiv=np.sum((Nv/kT)*fd12p((Ev_eff-EF)/kT),axis=0)
            return n,p,nderiv,pderiv
        else:
            return n,p

    @staticmethod
    def ionized_donor_density(mesh,EF,Ec,Ev,dopants,compute_derivs=True):
        Ndp=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']*(1/(1+d['g']*np.exp((EF-Ec+d["E"])/kT)))
           for d in dopants['Donor'].values())))
        Nam=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']*(1/(1+d['g']*np.exp((Ev+d["E"]-EF)/kT)))
           for d in dopants['Acceptor'].values())))

        if compute_derivs:
            Ndpderiv=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']*\
                    (d['g']/kT)*np.exp((EF-Ec+d["E"])/kT)/(1+d['g']*np.exp((EF-Ec+d["E"])/kT))**2
                for d in dopants['Donor'].values())))
            Namderiv=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']*\
                    (-d['g']/kT)*np.exp((Ev+d["E"]-EF)/kT)/(1+d['g']*np.exp((Ev+d["E"]-EF)/kT))**2
                for d in dopants['Acceptor'].values())))
            return Ndp,Nam,Ndpderiv,Namderiv
        else:
            return Ndp,Nam

    def solve(self,activation=1):
        m=self._mesh
        EF=m['EF']
        Ec=m['Ec']
        Ev=m['Ev']

        m['Ndp'],m['Nam'],m['Ndpderiv'],m['Namderiv']=\
            self.ionized_donor_density(EF,Ec,Ev,self._dopants,compute_derivs=True)

        m['n'],m['p'],m['nderiv'],m['pderiv']=self.carrier_density(EF,Ec,Ev,self._Nc,self._Nv,
            conduction_band_shifts=self._cDE,valence_band_shifts=self._vDE,compute_derivs=True)

        if activation!=1:
            for k in ['n','p','nderiv','pderiv','Ndp','Nam','Ndpderiv','Namderiv']:
                m[k]*=activation

        m['rho']=m['rho_pol']+q*(m['p']+m['Ndp']-m['n']-m['Nam'])
        m['rhoderiv']= q*(m['pderiv']+m['Ndpderiv']-m['nderiv']-m['Namderiv'])

