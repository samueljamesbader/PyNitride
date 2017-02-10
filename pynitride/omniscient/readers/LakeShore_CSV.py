# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:42:04 2016

@author: sam
"""
import numpy as np
import codecs
import re

def read(filename):
    with codecs.open(filename,'r','iso-8859-15') as f:
        data=[]
        colnames=[]
        metadata={}
        colnames=[re.sub(r'\s*\[.*','',x) for x in next(f).strip().replace("\t",",").split(",")]

        def toval(x):
            try: return float(x)
            except ValueError: return x
                
        # Is I-V curve export
        if "Resistance" in colnames:
            colnames=[{
                    "Current" : "I",
                    "Voltage" : "V",
                    "Resistance" : "R",
                    "Field": "B",
                    "Temperature": "T",
                }.get(c,c) for c in colnames]
            for l in f:
                if l.strip()=="": continue
                data+=[[toval(x) for x in l.strip().replace("\t",",").split(",")]]
            output=dict(zip(colnames,([np.array(c) for c in zip(*data)])))
        
        # Is Carrier export or identical
        elif sum([x.startswith("Hall Coeff.") for x in colnames]):
            colnames=np.array(colnames)
            Bavgcols=[re.sub(r"\s*\(.*","",x) for x in colnames[[0,1,2,8,9,10,20,21]]]
            Gavgcols=[re.sub(r"\s*\(.*","",x) for x in colnames[[0,3,4,5,11,12,20,21]]]
            Pavgcols=[re.sub(r"\s*\(.*","",x) for x in list(colnames[[0]])+["Hall Resistance"]+list(colnames[[20,21]])]
            rawcols=[re.sub(r"\s*\(.*","",x) for x in colnames[[0,13,14,15,16,17,18,19,20,21]]]
            
            Bavgdata=[]
            Gavgdata=[]
            Pavgdata=[]
            rawdata=[]
            
            for l in f:
                if l.strip()=="": continue
                vals=np.array([toval(x)
                    for x in l.strip().replace("\t",",").split(",")],dtype=object)
                
                # Bavg
                if vals[1]!="":
                    Bavgdata+=[list(vals[[0,1,2,8,9,10,20,21]])]
                # Gavg
                elif vals[3]!="":
                    Gavgdata+=[list(vals[[0,3,4,5,11,12,20,21]])]
                # Hall Res A
                elif vals[6]!="":
                    Pavgdata+=[['A']+list(vals[[0,6,20,21]])]
                # Hall Res A
                elif vals[7]!="":
                    Pavgdata+=[['B']+list(vals[[0,7,20,21]])]
                else:
                    rawdata+=[list(vals[[0,13,14,15,16,17,18,19,20,21]])]
                output={
                    'Bavg':
                    dict(zip(Bavgcols,
                        ([np.array(c) for c in zip(*Bavgdata)]))),
                    'Gavg':
                    dict(zip(Gavgcols,
                        ([np.array(c) for c in zip(*Gavgdata)]))),
                    'Pavg':
                    dict(zip(Pavgcols,
                        ([np.array(c) for c in zip(*Pavgdata)]))),
                    'raw':
                    dict(zip(rawcols,
                        ([np.array(c) for c in zip(*rawdata)])))}
        # Is HallMob export or identical
        elif sum([x.startswith("Hall Mobility") for x in colnames]):
            colnames=np.array(colnames)
            Bavgcols=[re.sub(r"\s*\(.*","",x) for x in colnames[[0,1,2,4,6,8,9,10,11,12,13,14,15]]]
            Gavgcols=[re.sub(r"\s*\(.*","",x) for x in colnames[[0,3,5,7,8,9,10,11,12,13,14,15]]]
            
            Bavgdata=[]
            Gavgdata=[]
            
            for l in f:
                if l.strip()=="": continue
                vals=np.array([toval(x)
                    for x in l.strip().replace("\t",",").split(",")],dtype=object)
                #print("so")
                #print(np.array([toval(x)
                #    for x in l.strip().replace("\t",",").split(",")],dtype=object))
                #print([x for x in vals])                
                #print([toval(x) for x in vals])
                # Bavg
                if vals[1]!="":
                    Bavgdata+=[list(vals[[0,1,2,4,6,8,9,10,11,12,13,14,15]])]
                # Gavg
                elif vals[3]!="":
                    Gavgdata+=[list(vals[[0,3,5,7,8,9,10,11,12,13,14,15]])]
                    
                output={
                    'Bavg':
                    dict(zip(Bavgcols,
                        ([np.array(c) for c in zip(*Bavgdata)]))),
                    'Gavg':
                    dict(zip(Gavgcols,
                        ([np.array(c) for c in zip(*Gavgdata)])))}
            
            #STOP
        else:
            output=dict([])
        output['_metadata']=metadata
        return output