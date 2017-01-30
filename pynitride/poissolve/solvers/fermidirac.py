# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 23:13:27 2017

@author: sam
"""

import re

import numpy as np
from pynitride.poissolve.maths.cfermidiracintegral import fd12, fd12p

from pynitride.paramdb import ParamDB
pmdb=ParamDB(units='neu')
pmdb.make_accessible(globals(),["k","hbar","e"]);q=e
kT=k*300
from pynitride.poissolve.mesh.functions import MaterialFunction, PointFunction, ConstantFunction


class FermiDirac3D():
    def __init__(self,mesh):
        self._mesh=mesh

        self._Nc,self._Nv=self.effective_dos_3d(mesh)
        self._cDE,self._vDE=self.band_edge_shifts(mesh)
        self._dopants=self.identifydopants(mesh)


    @staticmethod
    def identifydopants(mesh):
        dopants={'Donor':{},'Acceptor':{}}
        for d in [k[:-10] for k in mesh if k.endswith("ActiveConc")]:

            types=set(t for t in (l.material.get('dopant='+d+'.type',default=None) for l in mesh._layers) if t is not None)
            if len(types)>1: raise Exception(
                "Can't have one dopant be acceptor in one material and donor in another.  "\
                "You'll have to use two separate dopant names.  Sorry. ")
            if len(types)==1:
                dopants[list(types)[0]][d]={'conc':mesh[d+'ActiveConc']}
            else:
                print("No materials include {} as a dopant.".format(d))
        for doptype in dopants.keys():
            for d,v in dopants[doptype].items():
                v['E']=MaterialFunction(mesh,d+'.E',pos='point')
                v['g']=MaterialFunction(mesh,d+'.g',pos='point')
        mesh['Nd']=np.sum(d['conc'] for d in dopants['Donor'].values())\
            if len(dopants['Donor']) else ConstantFunction(mesh,0)
        mesh['Na']=np.sum(d['conc'] for d in dopants['Acceptor'].values()) \
            if len(dopants['Acceptor']) else ConstantFunction(mesh,0)
        return dopants

    @staticmethod
    def effective_dos_3d(mesh):

        # We'll have to confirm these formulae (factors of 2?) later...
        Nc=MaterialFunction(mesh,pos='point',prop=lambda mat:
            mat['electron.band.g']*(mat['electron.band.mdos']*kT/(2*np.pi*hbar**2))**(3/2))
        Nv=MaterialFunction(mesh,pos='point',prop=lambda mat:
            mat['hole.band.g']*(mat['hole.band.mdos']*kT/(2*np.pi*hbar**2))**(3/2))
        return Nc,Nv

    @staticmethod
    def band_edge_shifts(mesh):
        cDE=MaterialFunction(mesh,pos='point',prop=lambda mat: mat['electron.band.DE'])
        vDE=MaterialFunction(mesh,pos='point',prop=lambda mat: mat['hole.band.DE'])
        return cDE,vDE

    @staticmethod
    def carrier_density(EF,Ec,Ev,Nc,Nv,conduction_band_shifts=None,valence_band_shifts=None,compute_derivs=True):

        # Can save a copy and loop by checking if shifts are zero
        Ec_eff=Ec if conduction_band_shifts is None else Ec+conduction_band_shifts
        Ev_eff=Ev if valence_band_shifts is None else Ev-valence_band_shifts

        # Can save time by moving the summation inside the FD integral?
        n=np.sum(Nc*fd12((EF-Ec_eff)/kT),axis=0)
        p=np.sum(Nv*fd12((Ev_eff-EF)/kT),axis=0)

        if compute_derivs:
            nderiv=np.sum(-(Nc/kT)*fd12p((EF-Ec_eff)/kT),axis=0)
            pderiv=np.sum((Nv/kT)*fd12p((Ev_eff-EF)/kT),axis=0)
            return n,p,nderiv,pderiv
        else:
            return n,p

    @staticmethod
    def ionized_donor_density(mesh,EF,Ec,Ev,dopants,compute_derivs=True):

        # Tiwari Compound Semiconductor Devices pg31-32

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

    def solve(self,activation=1, quantum_band_shift=False):
        m=self._mesh
        EF=m['EF']
        Ec=m['Ec']
        Ev=m['Ev']

        m['Ndp'],m['Nam'],m['Ndpderiv'],m['Namderiv']=\
            self.ionized_donor_density(m,EF,Ec,Ev,self._dopants,compute_derivs=True)

        if quantum_band_shift:
            m['n'],m['p'],m['nderiv'],m['pderiv']=self.carrier_density(EF,Ec,Ev,self._Nc,self._Nv,
                conduction_band_shifts=self._cDE,valence_band_shifts=self._vDE,compute_derivs=True)
        else:
            m['n'],m['p'],m['nderiv'],m['pderiv']=self.carrier_density(EF,Ec,Ev,self._Nc,self._Nv,
               conduction_band_shifts=None,valence_band_shifts=None,compute_derivs=True)

        if activation!=1:
            for k in ['n','p','nderiv','pderiv','Ndp','Nam','Ndpderiv','Namderiv']:
                m[k]*=activation

        m['rho']=activation*m['rho_pol']+q*(m['p']+m['Ndp']-m['n']-m['Nam'])
        m['rhoderiv']= q*(m['pderiv']+m['Ndpderiv']-m['nderiv']-m['Namderiv'])

