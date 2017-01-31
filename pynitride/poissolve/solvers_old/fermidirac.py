# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 23:13:27 2017

@author: sam
"""

import re

import numpy as np

from pynitride.paramdb import ParamDB
pmdb=ParamDB(units='neu')
pmdb.make_accessible(globals(),["k","hbar","e"]);q=e
kT=k*300


