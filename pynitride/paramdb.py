from pynitride import ROOT_DIR
import os.path
import pint
import re
import scipy.constants as const
import numbers
import numpy as np

def parr(arr):
    if not len(arr): return arr
    if hasattr(arr[0],'units'):
        u=arr[0].units
        return np.array([(a/u).magnitude for a in arr])*u
    else:
        return np.array(arr)

class MultilevelDict():
    def __init__(self,dictionary={}):
        if isinstance(dictionary,MultilevelDict):
            self._dict=dictionary._dict
            self._index=dictionary._index
            self._shortformindex=dictionary._shortformindex
            self._nodes=dictionary._nodes
            self._extract=dictionary._extract
        else:
            self._dict=dictionary
            self._index=[]
            self._shortformindex=[]
            self._nodes=[]
            self._extract="value"

    def _subgetitem(self, cont, keyparts,extract, **constraints):
        if not len(keyparts):
            if isinstance(cont,Value):
                return getattr(cont,extract)
            elif isinstance(cont,dict):
                return list(cont.keys())
            elif isinstance(cont,list):
                return self._subgetitem(cont,["[:]"],extract,**constraints)
        k=keyparts[0]

        if isinstance(cont,list):
            if k.startswith("["):
                if not isinstance(cont,list):
                    raise Exception("Using [...] in a lookup expects an array parameter at that level.")
                if k=="[:]":
                    return parr([self._subgetitem(v,keyparts[1:],extract,**constraints) for v in cont])
                else:
                    return self._subgetitem(cont[int(k[1:-1])], keyparts[1:], extract,**constraints)
            else:
                return parr([self._subgetitem(v,keyparts,extract,**constraints) for v in cont])

        if k in cont:
            return self._subgetitem(cont[k], keyparts[1:], extract,**constraints)

        subs=list(s for s in cont.keys() if "=" in s)
        cvs=[s.split("=")[-1] for s in subs]
        if k in cvs:
            return self._subgetitem(cont[subs[cvs.index(k)]], keyparts[1:], extract,**constraints)

        cks=[s.split("=")[0] for s in subs]
        if k in cks:
            vals=[self._subgetitem(cont[keq], keyparts[1:], extract,**constraints)
                for keq in subs if keq.startswith(k+"=")]
            if len(keyparts)==1:
                return [keq.split("=")[1] for keq in subs if keq.startswith(k+"=")]
            return parr([v for v in vals if v is not None])

        for c in constraints:
            if c in cks:
                vals=[self._subgetitem(cont[c+"="+cv], keyparts, extract,**constraints) for cv in constraints[c] if c+'='+cv in cont]
                vals=[v for v in vals if v is not None]
                if len(vals): return vals[0]

        if '' in cvs:
            vals=[self._subgetitem(cont[keq], keyparts, extract,**constraints)
                  for keq in subs if keq.endswith("=")]
            vals=[v for v in vals if v is not None]
            if len(vals)>1:
                raise Exception("Key not uniquely specified.")
            if len(vals):
                return vals[0]
            else: return None


    def get(self,key,default=None,extract=None,**constraints):
        if isinstance(key,str): key=key.split(".")
        if extract is None: extract=self._extract
        v=self._subgetitem(self._dict,key,extract=extract,**constraints)
        return v if v is not None else default

    def __getitem__(self, key):
        v=self.get(key)
        if v is None: raise Exception("Multilevel key error: {:s}".format(key))
        return v

    def __setitem__(self,key,value):
        raise NotImplementedError
        keyparts=key.split(".")
        node=self._subgetitem(self._dict,keyparts[:-1])[keyparts[-1]]=Value(value,preparsed=True)

    def __getattr__(self, item):
        if item.startswith('__') and item.endswith('__'):
            raise AttributeError
        assert self!=self._dict, "WTF"
        return getattr(self._dict,item)

    def regenerate_index(self):
        self._index[:]=self._subregenerate_index(self._dict)
        self._shortformindex[:]=\
            [[ii.split("=")[-1] for ii in i if not ii.endswith('=')]
                  for i in self._subregenerate_index(self._dict)]

    def _subregenerate_index(self,cont):
        items=[]
        if isinstance(cont,dict):
            for k,v in cont.items():
                items+=[[k]+c for c  in self._subregenerate_index(v)]
        elif isinstance(cont,list):
            for k,v in enumerate(cont):
                items+=[["["+str(k)+"]"]+c for c in self._subregenerate_index(v)]
        else: items=[[]]
        return items

    def search(self,*keys,shortform=True):
        item_inds=[indi for indi,i in enumerate(self._shortformindex) if False not in [k in i for k in keys]]
        if shortform:
            items=[self._shortformindex[item_ind] for item_ind in item_inds]
        else:
            items=[self._index[item_ind] for item_ind in item_inds]
        return [".".join(i) for i in items]

class Value():
    def __init__(self, val, preparsed=False):
        if preparsed:
            self.raw=None
            self.parsed=val
        else:
            self.raw=val
            self.parsed=Value.parse(self.raw)
        self.neu=self.parsed.to_base_units().magnitude\
            if hasattr(self.parsed,'to_base_units')\
            else self.parsed

    @property
    def value(self):
        return self.parsed

    @staticmethod
    def parse(raw):
        # Python code
        if raw.startswith('`'):
            return eval(raw[1:-1])

        # string
        elif raw.startswith('"') or raw.startswith("'"):
            return raw[1:-1]

        # numerical expression to parse
        else:
            return ParamDB._ureg(raw)

def parse(val):
    v=ParamDB._ureg(val)
    return v.to_base_units().magnitude

def to_unit(val,unit):
    r""" Convert a number from PyNitride's internal units to any other units.

    Any number used in PyNitride or pulled from the parameter database is assumed to be in "nanoelectronic units", and
    this function provides conversion to other units for outputing readable results to a user.
    Note that this function does not and could not possibly ensure that your conversion is dimensionally valid, since
    the val input is just a number.  The user is responsible for knowing what val is (eg a distance, or an energy etc)
    and requesting sensible output dimensions.  The output dimensions are interpreted by Pint and a full list of allowed
    values (including units such as ``meter``, prefixed units such as ``meV``, more complex units like ``cm**-2`` and
    constants such as ``hbar``) is provided in the
    `Pint docs <https://github.com/hgrecco/pint/blob/master/pint/default_en.txt>`_.

    :param val: A numerical quantity which is in the internal unit system of PyNitride
    :param unit: The desired units for the result as a string (eg "cm**-2")
    :return: the number representing that quantity in the desired units.
    """
    return val/ParamDB._ureg(unit)

class ParamDB(MultilevelDict):

    # Enforce singleton pattern
    def __init__(self,globalDB=True,units='neu',temperature=300):

        # Make sure a static unit registry is loaded
        if not hasattr(ParamDB,'_ureg'):
            ParamDB._ureg=pint.UnitRegistry(system='neu')
            ParamDB._ureg.load_definitions(os.path.join(ROOT_DIR,"parameters","_system.txt"))

        # Make sure a global database exists
        if not hasattr(ParamDB,'_global'):
            # avoid infinite loop by setting the _global attribute before recursing
            ParamDB._global={}
            ParamDB._global=ParamDB()

            # Read in the default global parameter files
            ParamDB._global.read_file("_meta.txt", regenerate_index=False)
            for filename in ParamDB._global["meta.default global parameter files"]:
                ParamDB._global.read_file(filename, regenerate_index=False)
            ParamDB._global.regenerate_index()

        # If a global database is desired, we just need to share its data
        if globalDB:
            super(ParamDB,self).__init__(ParamDB._global)

        # If a local database is desired, we'll make it
        else:
            super(ParamDB,self).__init__(self)
            self.read_file("_meta.txt", regenerate_index=False)
            for filename in self["meta.default local parameter files"]:
                self.read_file(filename, regenerate_index=False)
            self.regenerate_index()

        if units=='Pint':
            self._extract='value'
        elif units=='neu':
            self._extract='neu'

    def clear(self):
        self._dict={}

    def read_file(self,filename, from_root=True, regenerate_index=True):
        if from_root:
            filename=os.path.join(ROOT_DIR,'parameters',filename)
        with open(filename) as f:
            firstline=next(f)
            if firstline.strip()==("PyNitride v2"):
                return self._read_PyNitride_paramfile(f, regenerate_index=regenerate_index)
            elif firstline.startswith("beta7"):
                return self._read_1DP_paramfile(f, regenerate_index=regenerate_index)

    def _read_PyNitride_paramfile(self,filehandle,regenerate_index=True):
        """ Reads the parameter structure from the PyNitride parameters file into this database.

        :param filehandle: the opened filehandle to read from (first line "PyNitride ..." has already been read out)
        """

        # Chain of nested subdictionaries to populate, and their indices, as we read through the file
        chain=[]

        # The subdictionary declared on the previous line of the file, and its indentation
        prev_dict=self._dict
        prev_indent=-1

        # Go through valid lines without comments
        for i,line in enumerate([l.rstrip().split('#')[0] for l in filehandle]):
            if line.strip()=="": continue

            # If this line is indented further than the previous, add the previous to the chain
            indent=len(line)-len(line.lstrip())
            if indent>prev_indent:
                # ... making sure the previous line actually declared a subdict (ie the previous was not a value line)
                if prev_dict is None:
                    raise Exception(
                        ("Syntax error: why is line {:d} of {:s} "+\
                        "indented further than the previous line?").format(i,filehandle.name))
                chain+=[[prev_dict,prev_indent]]

            # If this line is indented less than the previous, remove the previous from the chain
            else:
                while indent<=chain[-1][1]:
                    chain.pop()

            # Now get rid of indents
            line=line.strip()

            # If this is a value line
            if ":" in line and not line.endswith(":"):

                # Break at colon
                k,v=[p.strip() for p in line.split(":",maxsplit=1)]

                v=Value(v)

                # Add value to nested structure
                # If k is not '.', this is just adding it to the current subdictionary
                if k!='.': chain[-1][0][k]=v
                else: chain[-1][0]+=[v]

                # No new subdict to add
                prev_dict=None

            elif ":" in line and line.endswith(":"):
                k=line[:-1].strip()
                prev_dict=chain[-1][0][k]=[]

            # Otherwise, declare a new subdict
            else:
                # Normally, add line to subdict
                if line!='.':
                    if line not in chain[-1][0]:
                        chain[-1][0][line]={}
                    prev_dict=chain[-1][0][line]
                # But if the key is '.', add as blank dict
                else:
                    prev_dict={}
                    chain[-1][0]+=[prev_dict]

            prev_indent=indent

        if regenerate_index: self.regenerate_index()


    def make_accessible(self,destdict,variables=[],variablenames=None):
        if variablenames is None: variablenames=variables
        for v,vn in zip(variables,variablenames):
            destdict[vn]=getattr(Value(v),self._extract)



class Material():
    def __init__(self, matname, conditions=['relaxed'],pmdb=ParamDB()):
        self.matname=matname
        self.conditions=conditions
        self._pmdb=pmdb
    def __getitem__(self,key):
        val=self.get(key)
        if val is not None:
            return val
        else: raise Exception("Key not found: " + str(key) + " with constraints "+ str(self.conditions))
    def get(self,key,default=None):
        if isinstance(key,str): key=key.split('.');
        return self._pmdb.get(["material="+self.matname]+list(key),default=None,conditions=self.conditions)

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

ParamDB()







if __name__=="__main__":
    pmdb=ParamDB()
    print(pmdb['wurtzite.conventional.basis.[:].element'])

    pmdb2=ParamDB(units='neu')
    print(Material('GaN',pmdb=pmdb2)['dielectric'])
    #pmdb.read_file('VM2003.txt')
    #gan=Material("GaN")
    #gan['lattice','a']

    #print(pmdb._dict)
    #print(pmdb._index)
    #print(pmdb.search("GaN"))
