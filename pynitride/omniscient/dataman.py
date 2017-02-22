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
from functools import reduce
from datetime import date

from pynitride import config

class DataManager():
    def __init__(self,datadir=config['omniscient']['datadir']):
        self._datadir=datadir
        self._datatable=None
        
    def get_subprojects(self,proj="."):
        dirs=[x[0] for x in os.walk(os.path.join(self._datadir,proj),followlinks=True)]
        top=dirs[0]
        return [d[len(top)+1:] for d in dirs[1:]]

    def load_subproject(self,proj, expand=None, index='filename', extra={}):
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
            fields=[f for f in config['_headers'] if f not in config['DEFAULT']]
            try:
                headers=['filename','proj','mtype']+fields+['data']+list(extra.keys())
                evalers={f:eval(config['_headers'][f]) for f in fields}
            except Exception as e:
                print(e)
                raise Exception("Could not interpret [_headers] section in {}".format(indexpath))

            # Get the further fields
            furthers=[]
            for further in (s for s in config if s.startswith("_")):
                if further=="_headers":continue
                furthers+=[[config[further]['if'],config[further]['then']]]

            for mtype in config:
                if mtype.startswith('_'): continue
                if mtype=="DEFAULT": continue

                regex=re.compile(config[mtype]['nameregex'])
                reader=config[mtype]['reader'] if 'reader' in config[mtype] else mtype
                reader=importlib.import_module("pynitride.omniscient.readers."+reader)
                additional=eval(config[mtype]['additionalinfo']) if 'additionalinfo' in config[mtype] else {}

                pretable=[]
                for filename in filenames:

                    mo=re.match(regex,filename)
                    if mo:
                        #data=holder(reader.read(os.path.join(subdir,dirpath,filename)))
                        data=reader.read(os.path.join(subdir,dirpath,filename))
                        vals={name:evalers[name](mo.group(name)) for name in regex.groupindex.keys()}
                        vals.update(additional)
                        vals.update({'filename':os.path.join(dirpath,filename)[len(top)+1:] ,'proj':proj,'mtype':mtype, 'data': data})
                        vals.update(extra)

                        for further in furthers:
                            if eval(further[0],vals):
                                vals.update(eval(further[1],vals))
                        vals=[vals.get(c,None) for c in headers]
                        pretable+=[vals]
                pretable=dict(zip(headers,zip(*pretable)))
                temptable=pandas.DataFrame(pretable).set_index(index)
                dropdata=False
                if expand==True:
                    expand=list(reduce(set.union,(set(r.data.keys()) for r in temptable.itertuples())))
                    dropdata=True
                if isinstance(expand,list):
                    for c in expand:
                        temptable[c]=[r.data[c] for r in temptable.itertuples()]
                if dropdata:
                    del temptable['data']
                self._datatable=pandas.concat([self._datatable,temptable])


    @property
    def table(self):
        return self._datatable

    def __getitem__(self,key):
        return self._datatable.ix[key]

    def drop(self,*keys):
        for k in keys:
            del self._datatable[k]

    def copy(self,deep=True):
        dm=DataManager(self._datadir)
        dm._datatable=self._datatable.copy(deep=deep)
        return dm

    def expand_samplename(self, col='sample'):
        dates=[]
        if col in self._datatable:
            samp=self._datatable[col]
        elif self._datatable.index.name==col:
            samp=self._datatable.index
        else:
            raise Exception("Could not find "+str(col))
        for d in samp:
            try:
                y=2000+int(d[:2])
                m=int(d[2:4])
                d=int(d[4:6])
                dates+=[date(y,m,d)]
            except:
                dates+=[None]
        self._datatable['date']=dates

    def query(self,expr,inplace=False,**kwargs):
        if inplace:
            self._datatable.query(expr,inplace,**kwargs)
            return self
        else:
            dm=self.copy(deep=False)
            dm.query(expr,True,**kwargs)
            return dm

if __name__=="__main__":
    dm=DataManager("/home/sam/My_All/Cornell/Jena/ALL_DATA/")
    print(dm.get_projects())
    print(dm.get_subprojects("VO2 Modeling"))
    dt=dm.load_subproject("VO2 Modeling","T_Dep_Hall/2016_09_07")
    sam=dm._datatable
    