# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 10:36:06 2016

@author: sam
"""
import os
from configparser import ConfigParser, BasicInterpolation
import re
import pandas
import importlib

from pynitride import config

class DataManager():
    def __init__(self,datadir=config['omniscient']['datadir']):
        self._datadir=datadir
        self._datatable=None
        
    def get_subprojects(self,proj="."):
        dirs=[x[0] for x in os.walk(os.path.join(self._datadir,proj),followlinks=True)]
        top=dirs[0]
        return [d[len(top)+1:] for d in dirs[1:]]

    def load_subproject(self,proj):
        subdir=os.path.join(self._datadir,proj)
        assert os.path.isdir(subdir), "Subproject folder does not exist."
        top=None

        for dirpath,_,filenames in os.walk(subdir,followlinks=True):
            if top is None: top=dirpath
            indexpath=os.path.join(dirpath,"_meta.key")
            if not os.path.isfile(indexpath):
                continue
            else:
                print("Reading directory "+dirpath)
            config = ConfigParser(interpolation=BasicInterpolation())
            config.read(indexpath)
            config['DEFAULT']['DIR']=dirpath
            try:
                headers=['filename','proj','mtype']+config['_headers']['optional'].split(',')+['data']
            except:
                raise Exception("Could not find [_headers] optional section in {}".format(indexpath))

            # Get the further fields
            furthers=[]
            for further in (s for s in config if s.startswith("_")):
                if further=="_headers":continue
                furthers+=[[config[further]['if'],eval(config[further]['then'])]]

            for mtype in config:
                if mtype.startswith('_'): continue
                if mtype=="DEFAULT": continue

                regex=config[mtype]['nameregex']
                nameparts=config[mtype]['nameparts'].split(',')
                parttypes=config[mtype]['parttypes'].split(',')
                reader=config[mtype]['reader'] if 'reader' in config[mtype] else mtype
                reader=importlib.import_module("pynitride.omniscient.readers."+reader)
                additional=eval(config[mtype]['additionalinfo']) if 'additionalinfo' in config[mtype] else {}

                pretable=[]
                for filename in filenames:

                    mo=re.match(regex,filename)
                    if mo:
                        #data=holder(reader.read(os.path.join(subdir,dirpath,filename)))
                        data=reader.read(os.path.join(subdir,dirpath,filename))
                        vals={n:{"string": str, "int": int, "float": float}[pt](x) for x,n,pt in zip(mo.groups(),nameparts,parttypes)}
                        vals.update(additional)
                        vals.update({'filename':os.path.join(dirpath,filename)[len(top)+1:] ,'proj':proj,'mtype':mtype, 'data': data})

                        for further in furthers:
                            if eval(further[0],vals):
                                vals.update(further[1])
                        vals=[vals.get(c,None) for c in headers]
                        pretable+=[vals]
                pretable=dict(zip(headers,zip(*pretable)))
                self._datatable=pandas.concat([self._datatable,pandas.DataFrame(pretable).set_index('filename')])

    @property
    def table(self):
        return self._datatable
    
        
if __name__=="__main__":
    dm=DataManager("/home/sam/My_All/Cornell/Jena/ALL_DATA/")
    print(dm.get_projects())
    print(dm.get_subprojects("VO2 Modeling"))
    dt=dm.load_subproject("VO2 Modeling","T_Dep_Hall/2016_09_07")
    sam=dm._datatable
    