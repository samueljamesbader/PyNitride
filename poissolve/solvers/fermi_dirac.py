# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 23:13:27 2017

@author: sam
"""

import numpy as np
from poissolve.constants import kT,hbar,q
from poissolve.mesh_functions import PointFunction, MaterialFunction
from poissolve.maths.fermi_dirac_integral import fd12, fd12p
from poissolve.materials import _materials

class FermiDirac3D():
    def __init__(self,mesh,compute_dopants='GaN'):
        self._mesh=mesh
        mesh.add_function('n',PointFunction(mesh))
        mesh.add_function('p',PointFunction(mesh))

        # We'll have to confirm these formulae (factors of 2?) later... also one hole in GaN?
        self._Nc=MaterialFunction(mesh,pos='point',prop=lambda mat:
            mat['ladder','electron','g']*(mat['ladder','electron','mdos']*kT/(2*np.pi*hbar**2))**(3/2))
        self._Nv=MaterialFunction(mesh,pos='point',prop=lambda mat:
            mat['ladder']['hole']['g']*(mat['ladder']['hole']['mdos']*kT/(2*np.pi*hbar**2))**(3/2))

        self._nderiv=mesh.add_function('nderiv',PointFunction(mesh))
        self._pderiv=mesh.add_function('pderiv',PointFunction(mesh))
        self._Namderiv=PointFunction(mesh)
        self._Ndpderiv=PointFunction(mesh)
        self._rhoderiv=mesh.add_function('rhoderiv',PointFunction(mesh))
        self._rho_pol=0

    def _identifydopants(self):

        m=self._mesh
        for mat in (l.mat for l in m._layers):


        self._compute_dopants=compute_dopants
        self._Na=mesh['MgActiveConc']
        self._Nd=mesh['SiActiveConc']
        self._Nam=mesh['MgIonizedConc']
        self._Ndp=mesh['SiIonizedConc']
        

    def solve(self,activation=1):
        m=self._mesh
        EF=m['EF']
        Ec=m['Ec']
        Ev=m['Ev']
        n=m['n']
        nderiv=self._nderiv
        pderiv=self._pderiv
        p=m['p']
        Ndp=self._Ndp
        Nam=self._Nam
        Namderiv=self._Namderiv
        Ndpderiv=self._Ndpderiv
        rhoderiv=self._rhoderiv
        rho=m['rho']

        n[:]=self._Nc*fd12((EF-Ec)/kT)
        p[:]=self._Nv*fd12((Ev-EF)/kT)

        if self._compute_dopants:
            d=_materials[self._compute_dopants]['dopants']['Si']
            Ndp[:]=self._Nd*(1/(1+d['g']*np.exp((EF-Ec+d["E"])/kT)))
            Ndpderiv[:]=self._Nd*(
                (d['g']/kT)*np.exp((EF-Ec+d["E"])/kT)/
                (1+d['g']*np.exp((EF-Ec+d["E"])/kT))**2)
            d=_materials[self._compute_dopants]['dopants']['Mg']
            Nam[:]=self._Na*(1/(1+d['g']*np.exp((Ev+d["E"]-EF)/kT)))
            Namderiv[:]=self._Na*(
                (-d['g']/kT)*np.exp((Ev+d["E"]-EF)/kT)/
                (1+d['g']*np.exp((Ev+d["E"]-EF)/kT))**2)

        nderiv[:]=-(self._Nc/kT)*fd12p((EF-Ec)/kT)
        pderiv[:]= (self._Nv/kT)*fd12p((Ev-EF)/kT)

        temp_rho=(self._rho_pol+q*(p+Ndp)-q*(n+Nam))

        rho[:]=activation*temp_rho
        rhoderiv[:]=\
            activation*(q*(pderiv+Ndpderiv)-q*(nderiv+Namderiv))
