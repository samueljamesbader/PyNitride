# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:42:04 2016

@author: sam
"""
import numpy as np

def read(filename):
    with open(filename) as f:
        data=[]
        colnames=[]
        metadata={}
        for l in f:
            #print(l)
            if l.startswith("TestParameter"):
                metadata[l.split(',')[1].strip()]=\
                    [v.strip() for v in l.split(',')[2:]]
            if l.startswith("DataName"):
                colnames=[n.strip() for n in l.split(',')[1:]]
            elif l.startswith("DataValue"):
                data+=[[float(v.strip()) if v.strip()!="" else np.NaN for v in l.split(',')[1:]]]
        #if(len(data)==0): data=[[]]*len(colnames)
        output=dict(zip(colnames,([np.array(c) for c in zip(*data)])))
        output['_metadata']=metadata
        #output['filename']=filename
        return output