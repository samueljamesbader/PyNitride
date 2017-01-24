from pynitride import ROOT_DIR
import os.path
import pint
import re
import scipy.constants as const

class MultilevelDict():
    def __init__(self,dictionary):
        self._dict=dictionary

    def __getitem__(self, key, **constraints):
        if isinstance(key, str): key=[key]
        try:
            d=self._dict
            for k in key:
                for c in constraints.keys():
                    if c in d:
                        d=d[c][constraints[c]]
                d=d[k]
            return d
        except:
            raise Exception("Multilevel key error: " + str(key))

    def get(self,param,default=None,**constraints):
        try:
            return self.__getitem__(param,**constraints)
        except:
            return default


    def __setitem__(self,key,value):
        if isinstance(key, str): key=[key]
        d=self._dict
        for k in key[:-1]:
            if k not in d:
                d[k]={}
            d=d[k]
        d[key[-1]]=value


    def __getattr__(self, item):
        if item.startswith('__') and item.endswith('__'):
            raise AttributeError

        return getattr(self._dict,item)


class ParamDB(MultilevelDict):
    def __init__(self, system='mks', make_global=False, load_files=['VM2003.txt']):
        if not hasattr(ParamDB,"_ureg"):
            ParamDB._ureg=pint.UnitRegistry(system=system)
            #ParamDB._ureg.load_definitions(os.path.join(ROOT_DIR,"parameters","constants.txt"))
        assert ParamDB._ureg.default_system==system, "Unit systems error"

        self._dict={}
        if make_global: ParamDB._global=self
        for f in load_files: self.read_file(f)

    @staticmethod
    def get_global(*args,**kwargs):
        if hasattr(ParamDB,"_global"):
            return ParamDB._global
        else:
            return ParamDB(*args,make_global=True,**kwargs)

    def read_file(self,filename,from_root=True):
        if from_root:
            filename=os.path.join(ROOT_DIR,'parameters',filename)
        with open(filename) as f:
            k=[]
            indents=[[-1,[]]]
            for line in [l.rstrip().split('#')[0] for l in f]+['#']:
                if line.strip()=="": continue
                items_on_line=[k2 for k2 in [k.strip() for k in line.split(':')] if k2!=""]

                indent=len(line)-len(line.lstrip())

                # a line at same or lesser indent is the signal to add the previously stored list to dict
                if indent <= indents[-1][0]:
                    self[indents[-1][1][:-1]]=value_parser(indents[-1][1][-1])
                while indent <= indents[-1][0]:
                    indents.pop()
                indents.append([indent,indents[-1][1]+items_on_line])

def value_parser(val, err_on_fail=False):
    ParamDB.get_global()
    try: return float(val)
    except: pass
    try: return ParamDB._ureg(val).to_base_units().magnitude
    except: pass
    if err_on_fail: raise Exception("Could not parse "+val)
    return val

def to_unit(val,unit):
    return val/value_parser(unit,err_on_fail=True)

hbar=to_unit(const.hbar,"J s")
m_e=to_unit(const.electron_mass,"J s")


class Material():
    def __init__(self, matname, pdb=None,conditions=['relaxed','default']):
        self.matname=matname
        self.conditions=conditions
        self._pdb=pdb if pdb else ParamDB.get_global()
    def __getitem__(self,key):
        if isinstance(key, str): key=[key]
        for condition in self.conditions:
            val=self._pdb.get(["material",self.matname]+list(key),default=None,conditions=condition)
            if val is not None:
                return val
        raise Exception("Key not found: " + str(key) + " with constraints "+ str(self.conditions))
    def get(self,key,default=None):
        try:
            return self.__getitem__(key)
        except:
            return default

if __name__=="__main__":

    pdb=ParamDB(make_global=True)
    pdb.read_file('VM2003.txt')
    gan=Material("GaN")
    gan['lattice','a']
    print(pdb._dict)
