# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:42:04 2016

@author: sam
"""
import numpy as np
import re

def read(filename):
    with open(filename) as f:
        data=[]
        l=next(f)
        colnames=[n.strip() for n in l.split('\t')]
        for l in f:
            vals=[]
            for v in l.split('\t'):
                try:
                    vals+=[float(v)]
                except:
                    vals+=[np.NaN]
            data+=[vals]
        preoutput=dict(zip(colnames,([np.array(c) for c in zip(*data)])))
        dcols=[np.array(c) for c in zip(*data)]

        sweeps=[]
        for i,col in enumerate(colnames):
            mo=re.match(r'([^\(]+)\((\d+)\)',col)
            if mo:
                s=int(mo.group(2))
                if len(sweeps)<s:
                    sweeps+=[{}]
                sweeps[s-1][mo.group(1)]=dcols[i]

        return sweeps
