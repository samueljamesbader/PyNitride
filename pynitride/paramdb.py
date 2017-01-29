from pynitride import ROOT_DIR
import os.path
import pint
import re
import scipy.constants as const
import numbers

class MultilevelDict():
    def __init__(self,dictionary):
        self._dict=dictionary

    @classmethod
    def _subgetitem(cls,dic,keyparts,**constraints):
        if not len(keyparts): return dic
        k=keyparts[0]

        if k==":":
            return [cls._subgetitem(v,keyparts[1:],**constraints) for v in dic.values()]

        if k in dic:
            return cls._subgetitem(dic[k],keyparts[1:],**constraints)

        subs=list(dic.keys())
        cvs=[s.split("=")[-1] for s in subs if "=" in s]
        if k in cvs:
            return cls._subgetitem(dic[subs[cvs.index(k)]],keyparts[1:],**constraints)

        cks=[s.split("=")[0] for s in subs if "=" in s]
        if k in cks:
            vals=[cls._subgetitem(dic[subs[keq]],keyparts[1:],**constraints)
                for keq in subs if keq.startswith(k+"=")]
            return [v for v in vals if v is not None]

        for c in constraints:
            if c in dic:
                return [cls._subgetitem(dic[cv],keyparts[1:],**constraints) for cv in constraints[c]]

        if '.' in cvs:
            vals=[cls._subgetitem(dic[subs[eqd]],keyparts,**constraints)
                  for keq in subs if keq.endswith("=.")]
            return cls._subgetitem(dic[subs[cvs.index(k)]],keyparts[1:],**constraints)


    def get(self,key,default=None,**constraints):
        v=self._subgetitem(self._dict,key.split("."))
        return v if v is not None else default

    def __getitem__(self, key):
        v=self._subgetitem(self._dict,key.split("."))
        if v is None: raise Exception("Multilevel key error: {:s}".format(key))
        return v

    def get(self, key,default=None,**constraints):
        v=self._subgetitem(self._dict,key.split("."),**constraints)
        return v if v is not None else default

    def __setitem__(self,key,value):
        keyparts=key.split(".")
        self._subgetitem(self._dict,keyparts[:-1])[keyparts[-1]]=value

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

            self.read_file("_meta.txt")
            for filename in self["meta.default parameter files"]:
                self.read_file(filename)
        return ParamDB._self

    def __init__(self): pass

    def clear(self):
        self._dict={}

    def read_file(self,filename,from_root=True):
        if from_root:
            filename=os.path.join(ROOT_DIR,'parameters',filename)
        with open(filename) as f:
            firstline=next(f)
            if firstline.strip()==("PyNitride v2"):
                return self._read_PyNitride_paramfile(f)
            elif firstline.startswith("beta7"):
                return self._read_1DP_paramfile(f)

    def _read_PyNitride_paramfile(self,filehandle):
        """ Reads the parameter structure from the PyNitride parameters file into this database.

        :param filehandle: the opened filehandle to read from (first line "PyNitride ..." has already been read out)
        """

        # Chain of nested subdictionaries to populate as we read through the file
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
                chain+=[prev_dict]

            # If this line is indented less than the previous, remove the previous from the chain
            elif indent<prev_indent:
                chain.pop()

            # Now get rid of indents
            line=line.strip()

            # If this is a value line
            if ":" in line and not line.endswith(":"):

                # Break at colon
                k,v=[p.strip() for p in line.split(":",maxsplit=1)]

                # Python code
                if v.startswith('`'):
                    v=eval(v[1:-1])

                # string
                elif v.startswith('"') or v.startswith("'"):
                    v=v[1:-1]

                # numerical expression to parse
                else:
                    v=parse(v)

                # Add value to nested structure
                # If k is not '.', this is just adding it to the current subdictionary
                try:
                    if k!='.': chain[-1][k]=v
                    else: chain[-1]+=[v]
                except Exception as e:
                    what

                # No new subdict to add
                prev_dict=None

            elif ":" in line and line.endswith(":"):
                k=line[:-1].strip()
                prev_dict=chain[-1][k]=[]

            # Otherwise, declare a new subdict
            else:
                # Normally, add line to subdict
                if line!='.':
                    prev_dict=chain[-1][line.strip()]={}
                # But if the key is '.', add as blank dict
                else:
                    prev_dict={}
                    chain[-1]+=[prev_dict]


            prev_indent=indent


    def _read_1DP_paramfile(self,filehandle):

        for line in filehandle:
            mo=re.match(r"^(\w+)\s+binary\s+\w+",line.strip())
            if mo:
                matname=mo.groups(0)[0]

                tmp={}
                next(filehandle) # skip mystery zeros line in materials file
                for line in filehandle:
                    mo=re.match(r"(\w+)=([\d\*eE\+\-\.Temp]+)",line)
                    if mo is None: break
                    try:
                        tmp[mo.groups()[0]]=eval("(lambda Temp: "+mo.groups()[1].replace('^','**')+")("+str(to_unit(T,'K'))+")")
                        if matname=="GaN" or matname=="AlN":
                            if mo.groups()[0]=='pol':
                                print(mo.groups(),tmp[mo.groups()[0]])
                    except Exception as e:
                        print(e)
                        print(mo.groups())
                        import numpy as np
                        tmp[mo.groups()[0]]=np.NaN

                import scipy.constants as const
                self['material',matname,'conditions','default']=dict(
                    bands=dict(
                        Eg=tmp['eg'] *eV,
                        DEc=tmp['dec'] *eV,
                        barrier=dict(),
                        electron=dict(
                            Gamma=dict(g=2*tmp['val'],mzs=tmp['me']*m_e,mxys=tmp['me']*m_e,mdos=tmp['me']*m_e,DE=0)),
                        hole=dict(
                            HH=dict(g=2,mzs=tmp['mh']*m_e,mxys=tmp['mh']*m_e,mdos=tmp['mh']*m_e,DE=0),
                            LH=dict(g=2,mzs=tmp['mlh']*m_e,mxys=tmp['mlh']*m_e,mdos=tmp['mlh']*m_e,DE=0),
                            SO=dict(g=2,mzs=tmp['mhso']*m_e,mxys=tmp['mhso']*m_e,mdos=tmp['mhso']*m_e,DE=0))),
                    dielectric=dict(eps=tmp['er']*epsilon_0),dopants=dict(
                        Donor=dict(type='Donor',E=tmp['ed']*eV, g=2),
                        Acceptor=dict(type='Acceptor',E=tmp['ea']*eV, g=4),
                        DeepDonor=dict(type='Donor',E=tmp['edd']*eV, g=2),
                        DeepAcceptor=dict(type='Acceptor',E=tmp['eda']*eV, g=4)),
                    polarization=dict(Ptot=-tmp['pol']/const.elementary_charge/cm**2))


def parse(val):
    pdb=ParamDB()
    v=pdb._ureg(val)
    if isinstance(v,numbers.Number): return v
    else: return v.to_base_units().magnitude

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
    pdb=ParamDB()
    return val/pdb.parse(unit,err_on_fail=True)


#pdb=ParamDB()
#convenient_constants=["hbar","c","m_e","angstrom","nm","um","mm","cm","mV","V","kV","MV","meV","eV","keV","MeV","epsilon_0"]
#for const in convenient_constants:
#    globals()[const]=parse(const,err_on_fail=True)
#
#convenient_constants+=['q','kT']
#q=parse('e',err_on_fail=True)
#T=300
#kT=parse('k',err_on_fail=True)*T

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
    #gan=Material("GaN")
    #gan['lattice','a']
    print(pdb._dict)
