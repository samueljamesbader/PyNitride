# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:18:47 2017

@author: sam
"""

import scipy.constants as const
#from numpy import pi

#pi=np.pi
q=1.0
k=const.Boltzmann/const.elementary_charge
T=300.0
kT=k*T
hbar=const.hbar/const.elementary_charge/1e-15
m0=const.electron_mass/const.elementary_charge*1e12
eps_0=const.epsilon_0/const.elementary_charge/1e9

cm=1.0e7
nm=1.0
eV=1
MV=1e6