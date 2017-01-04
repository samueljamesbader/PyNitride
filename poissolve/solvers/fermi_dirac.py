# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 23:13:27 2017

@author: sam
"""

import numpy as np
from poissolve.constants import kT,hbar,q
from poissolve.mesh_functions import PointFunction, MaterialFunction
from poissolve.maths.fermi_dirac_integral import fd12, fd12p
from poissolve.materials import import _materials

class FermiDirac3D():
    def __init__(self,mesh,compute_dopants='GaN'):
        self._mesh=mesh
        mesh.add_function('n',PointFunction(mesh))
        mesh.add_function('p',PointFunction(mesh))

        # We'll have to confirm these formulae later... also one hole in GaN?
        self._Nc=MaterialFunction(mesh,pos='point',prop=lambda mat:
            mat['ladder','electron','g']*(mat['ladder','electron','mdos']*kT/(2*np.pi*hbar**2))**(3/2)).array
        self._Nv=MaterialFunction(mesh,pos='point',prop=lambda mat:
            mat['ladder']['hole']['g']*(mat['ladder']['hole']['mdos']*kT/(2*np.pi*hbar**2))**(3/2)).array

        self._compute_dopants=compute_dopants
        self._Na=mesh.get_function('MgActiveConc')
        self._Nd=mesh.get_function('SiActiveConc')
        self._Nam=mesh.get_function('MgIonizedConc')
        self._Ndp=mesh.get_function('SiIonizedConc')
        
        self._nderiv=mesh.add_function('nderiv',PointFunction(mesh))
        self._pderiv=mesh.add_function('pderiv',PointFunction(mesh))
        self._Namderiv=PointFunction(mesh)
        self._Ndpderiv=PointFunction(mesh)
        self._rhoderiv=PointFunction(mesh)
        
        
    def solve(self,damp=0,clamp=.001,derivs=False,activation=1):
        m=self._mesh

        m['n'].array=self._Nc*fd12((m['EF']-m['Ec'])/kT)
        m['p'].array=self._Nv*fd12((m['Ev']-m['EF'])/kT)

        if self._compute_dopants:
            #print(" ASSUMING DOPANTS ARE IN ALN")
            d=_materials[self._compute_dopants]['dopants']['Si']
            self._Ndp.array=self._Nd.array*(1/(1+d['g']*np.exp((self._EF-self._Ec+d["E"])/kT)))
            self._Ndpderiv.array=self._Nd.array*(
                (-d['g']/kT)*np.exp((self._EF-self._Ec+d["E"])/kT)/
                (1+d['g']*np.exp((self._EF-self._Ec+d["E"])/kT))**2)
            d=_materials[self._compute_dopants]['dopants']['Mg']
            self._Nam.array=self._Na.array*(1/(1+d['g']*np.exp((self._Ev+d["E"]-self._EF)/kT)))
            self._Namderiv.array=self._Na.array*(
                (d['g']/kT)*np.exp((self._Ev+d["E"]-self._EF)/kT)/
                (1+d['g']*np.exp((self._Ev+d["E"]-self._EF)/kT)))
            
        
        temp_rho=(self._rho_pol+q*(self._p.array+self._Ndp.array)-q*(self._n.array+self._Nam.array))
        
        #self._rho.array=temp_rho*activation
        
        
        #print("Charge change: {:.2g}".format(np.sqrt(np.sum((temp_rho-self._rho.array)**2))))
        
        
        if True:
            self._nderiv.array=-(self._Nc/kT)*fd12p((self._EF-self._Ec)/kT)
            self._pderiv.array= (self._Nv/kT)*fd12p((self._Ev-self._EF)/kT)
            #from IPython.core.debugger import Tracer;Tracer()()
            self._rho.array=activation*temp_rho
            self._rhoderiv.array=\
                activation*(q*(self._pderiv.array+self._Ndpderiv.array)-q*(self._nderiv.array+self._Namderiv.array))
        else:

            charginess_prev=np.sum(np.abs(self._rho.array)*self._mesh._dzp)
            charginess_temp=np.sum(np.abs(temp_rho)*self._mesh._dzp)

            cc=np.abs(charginess_temp-charginess_prev)
            if cc<clamp:
                self._rho.array=(damp)*self._rho.array+(1-damp)*temp_rho
                print('clamping charge')
            else:
                self._rho.array=(1-clamp/cc)*self._rho.array+(clamp/cc)*temp_rho