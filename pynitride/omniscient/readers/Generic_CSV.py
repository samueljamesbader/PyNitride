# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:42:04 2016

@author: sam
"""
import numpy as np

def read(filename):
    with open(filename) as f:
        data=[]
        l=next(f)
        colnames=[n.strip() for n in l.split(',')]
        for l in f:
            vals=[]
            for v in l.split(','):
                try:
                    vals+=[float(v)]
                except:
                    vals+=[np.NaN]
            data+=[vals]
        output=dict(zip(colnames,([np.array(c) for c in zip(*data)])))
        output['_metadata']={}
        #output['filename']=filename
        return output
