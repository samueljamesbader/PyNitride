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

    def get_subprojects(self,proj="."):
        dirs=[x[0] for x in os.walk(os.path.join(self._datadir,proj),followlinks=True)]
        top=dirs[0]
        return [d[len(top)+1:] for d in dirs[1:]]

    def load_subproject(self,proj, expand=None, index='filename', extra={}, datatable=None):
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
                if not len(pretable): continue
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
                datatable=pandas.concat([datatable,temptable])
        return datatable

