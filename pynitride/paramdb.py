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

    # Enforce singleton pattern
    def __new__(cls):
        if not hasattr(ParamDB,'_self'):
            self=ParamDB._self=super().__new__(cls)
            super(ParamDB,self).__init__({})
            self._ureg=pint.UnitRegistry(system='neu')
            self._ureg.load_definitions(os.path.join(ROOT_DIR,"parameters","_system.txt"))

            self.read_file("_meta")
            print(self._dict)
            for filename,yn in self["meta","default parameter files"].items():
                if yn=="yes":
                    self.read_file(filename)
        return ParamDB._self

    def __init__(self): pass

    def clear(self):
        self._dict={}

    def read_file(self,filename,from_root=True):
        if from_root:
            filename=os.path.join(ROOT_DIR,'parameters',filename+".txt")
        with open(filename) as f:
            k=[]
            indents=[[-1,[]]]
            for line in [l.rstrip().split('#')[0] for l in f]+['#']:
                if line.strip()=="": continue
                items_on_line=[k2 for k2 in [k.strip() for k in line.split(':')] if k2!=""]

                indent=len(line)-len(line.lstrip())

                # a line at same or lesser indent is the signal to add the previously stored list to dict
                if indent <= indents[-1][0]:
                    self[indents[-1][1][:-1]]=parse(indents[-1][1][-1])
                while indent <= indents[-1][0]:
                    indents.pop()
                indents.append([indent,indents[-1][1]+items_on_line])

    @staticmethod
    def parse(val, err_on_fail=False):
        pdb=ParamDB()

        if val=="[]": return []

        try: return float(val)
        except: pass
        try: return pdb._ureg(val).to_base_units().magnitude
        except: pass
        if err_on_fail: raise Exception("Could not parse "+val)
        return val

    @staticmethod
    def to_unit(val,unit):
        pdb=ParamDB()
        return val/pdb.parse(unit,err_on_fail=True)


parse=ParamDB.parse
to_unit=ParamDB.to_unit

pdb=ParamDB()
convenient_constants=["hbar","c","m_e","angstrom","nm","um","mm","cm","mV","V","kV","MV","meV","eV","keV","MeV","epsilon_0"]
for const in convenient_constants:
    globals()[const]=parse(const,err_on_fail=True)

convenient_constants+=['q','kT']
q=parse('e',err_on_fail=True)
T=300
kT=parse('k',err_on_fail=True)*T

class Material():
    def __init__(self, matname, conditions=['relaxed','default']):
        self.matname=matname
        self.conditions=conditions
        self._pdb=ParamDB()
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

    # http://stackoverflow.com/a/25176504/2081118
    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return self._matname==other._matname
        return NotImplemented
    def __neq__(self,other):
        if isinstance(other,self.__class__):
            return self._matname!=other._matname
        return NotImplemented
    def __hash__(self):
        return hash(self._matname)









if __name__=="__main__":

    pdb=ParamDB()
    pdb.read_file('VM2003.txt')
    gan=Material("GaN")
    gan['lattice','a']
    print(pdb._dict)
