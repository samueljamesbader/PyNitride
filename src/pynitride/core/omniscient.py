# This script defines a Brain class that can store complex nested structures where values have units
# It's built to read from the custom pynitride material parameter files
# It's basically just the plumbing for pynitride's parameter database

from collections import OrderedDict
import numpy as np
import os.path
from pint import UnitRegistry

ThrowExceptionIfNotFound=object()

class Brain():

    def __init__(self,folder,dictionary=None):
        self._folder=folder

        #if isinstance(dictionary, Brain):
        #    self._dict=dictionary._dict
        #    self._index=dictionary._index
        #    self._shortformindex=dictionary._shortformindex
        #    self._nodes=dictionary._nodes
        #else:
        #    if dictionary is None: dictionary=OrderedDict()
        if 1:
            self._dict=OrderedDict()
            self._index=[]
            self._shortformindex=[]
            self._nodes=[]


    def _subgetitem(self, cont, keyparts, **constraints):
        """ Helper function for __call__().

        Recursive procedure: for each call of _subgetitem, look in the current container for the next keypart.

        Args:
            cont: a container (eg dictionary, list, or Brain.Value).
            keyparts (list): array of keys left to employ in digging for the desired value.
            constraints: a list of key-value pairs which may be employed at any level in the recursion
                when the current keypart doesn't offer a way forward.
        """

        # If there are no keyparts left to use and...
        if not len(keyparts):

            # ... we have a Value, we're done: return it
            if isinstance(cont,Brain.Value):
                return cont

            # ... we have a dictionary, then return a list of the keys for convenience
            elif isinstance(cont,dict):
                return list(cont.keys())

            # ... we have a list, then get all the elements of it
            elif isinstance(cont,list):
                return self._subgetitem(cont,["[:]"],**constraints)

        # Grab the current keypart
        k=keyparts[0]

        # If it's a list
        if isinstance(cont,list):

            # Check if the current keypart is a list index
            if k.startswith("["):

                # If it's "[:]", go deeper with each list index
                if k=="[:]":
                    vals=[self._subgetitem(v,keyparts[1:],**constraints) for v in cont]
                    if None in vals: return None
                    else: return [v.get() for v in vals]

                # Otherwise, just go deeper with the given index
                else:
                    return self._subgetitem(cont[int(k[1:-1])], keyparts[1:], **constraints)

            # If the current keypart is not a list index, then go deeper with each element of the list
            else:
                vals=[self._subgetitem(v,keyparts[1:],**constraints) for v in cont]
                if None in vals: return None
                else: return [v.get() for v in vals]

        # If the keypart starts with "[", but we don't have a list, complain
        elif k.startswith("["):
            raise Exception("Using [...] in a lookup expects an array parameter at that level.")

        # If the keypart is a key of the current container, go deeper into that branch
        if k in cont:
            return self._subgetitem(cont[k], keyparts[1:], **constraints)

        # If the keypart is the Y of a key in that container of the form X=Y, go deeper into that branch
        subs=list(s for s in cont.keys() if "=" in s)
        cvs=[s.split("=")[-1] for s in subs]
        if k in cvs:
            return self._subgetitem(cont[subs[cvs.index(k)]], keyparts[1:], **constraints)

        # If the keypart is an X for a key of the form X=Y in this container
        cks=[s.split("=")[0] for s in subs]
        if k in cks:

            # If this is the last keypart, then, for convenience return the possible Y's
            if len(keyparts)==1:
                return [keq.split("=")[1] for keq in subs if keq.startswith(k+"=")]

            # Otherwise dig deeper for each and...
            vals=[self._subgetitem(cont[keq], keyparts[1:], **constraints)
                  for keq in subs if keq.startswith(k+"=")]

            # ...if the results are empty just return None and a higher up level will handle
            # (maybe another branch has the desired value)
            vals=[v for v in vals if v is not None]
            if len(vals)==0: return None

            # ...if the results are ambiguous, complain
            elif len(vals)>1: raise Exception("Results ambiguous because of keypart "+k)

            # ...if there's just one result, return it.
            else: return vals[0]

        # If there are constrains which could help choose at this point
        for c in constraints:
            if c in cks:

                # Go deeper for each matching constraint
                vals=[self._subgetitem(cont[c+"="+cv], keyparts, **constraints) for cv in constraints[c] if c+'='+cv in cont]

                # ...if the results are empty just return None and a higher up level will handle
                # (maybe another branch has the desired value)
                vals=[v for v in vals if v is not None]
                if len(vals)==0: return None

                # ...if the results are ambiguous, complain
                elif len(vals)>1: raise Exception("Results ambiguous")

                # ...if there's just one result, return it.
                else: return vals[0]

        # If there is a fallback for this level, try it
        if '' in cvs:
            vals=[self._subgetitem(cont[keq], keyparts, **constraints)
                  for keq in subs if keq.endswith("=")]
            # ...if the results are empty just return None and a higher up level will handle
            # (maybe another branch has the desired value)
            vals=[v for v in vals if v is not None]
            if len(vals)==0: return None

            # ...if the results are ambiguous, complain
            elif len(vals)>1: raise Exception("Results ambiguous")

            # ...if there's just one result, return it.
            else: return vals[0]

    def __call__(self,key,default=ThrowExceptionIfNotFound,**constraints):
        if isinstance(key,str): key=key.split(".")
        v=self._subgetitem(self._dict,key,**constraints)
        if v is None:
            if default is ThrowExceptionIfNotFound:
                raise Exception('Key "{}" not found'.format(".".join(key)))
            return default
        if isinstance(v,Brain.Value):
            return v.get()
        else:
            return v

    def __getitem__(self, key):
        return self.__call__(key)

    #def __setitem__(self,key,value):
    #    raise NotImplementedError
    #    keyparts=key.split(".")
    #    node=self._subgetitem(self._dict,keyparts[:-1])[keyparts[-1]]=Value(value,preparsed=True)

    #def __getattr__(self, item):
    #    if item.startswith('__') and item.endswith('__'):
    #        raise AttributeError
    #    assert self!=self._dict, "WTF"
    #    return getattr(self._dict,item)

    #def regenerate_index(self):
    #    self._index[:]=self._subregenerate_index(self._dict)
    #    self._shortformindex[:]= \
    #        [[ii.split("=")[-1] for ii in i if not ii.endswith('=')]
    #         for i in self._subregenerate_index(self._dict)]

    #def _subregenerate_index(self,cont):
    #    items=[]
    #    if isinstance(cont,dict):
    #        for k,v in cont.items():
    #            items+=[[k]+c for c  in self._subregenerate_index(v)]
    #    elif isinstance(cont,list):
    #        for k,v in enumerate(cont):
    #            items+=[["["+str(k)+"]"]+c for c in self._subregenerate_index(v)]
    #    else: items=[[]]
    #    return items

    #def search(self,*keys,shortform=True):
    #    item_inds=[indi for indi,i in enumerate(self._shortformindex) if False not in [k in i for k in keys]]
    #    if shortform:
    #        items=[self._shortformindex[item_ind] for item_ind in item_inds]
    #    else:
    #        items=[self._index[item_ind] for item_ind in item_inds]
    #    return [".".join(i) for i in items]

    #def quantity(self,*args):
    #    # If it's a single array, and not made of numeric or Pint quantity elements, just return as is
    #    if len(args)==1 and hasattr(args[0],'__iter__') and not isinstance(args[0],str):
    #        if args[0]==[]: return np.array([])
    #        elif not (isinstance(args[0][0],numbers.Number) or hasattr(args[0][0],"units")):
    #            return args[0]

    #    if self._units=='neu':
    #        if len(args)==1 and isinstance(args[0],str):
    #            if "," in args[0]:
    #                return [self.quantity(s) for s in args[0].split(',')]
    #        elif len(args)==1 and hasattr(args[0],'__iter__'):
    #            return np.array([self.quantity(a) for a in args[0]])
    #        return ParamDB._ureg.Quantity(*args).to_base_units().magnitude
    #    elif self._units=='Pint':
    #        if len(args)==1 and isinstance(args[0],str):
    #            if "," in args[0]:
    #                return [self.quantity(s) for s in args[0].split(',')]
    #        elif len(args)==1 and hasattr(args[0],'__iter__'):
    #            lst=args[0]
    #            if not len(lst):
    #                return np.array([])
    #            elif hasattr(lst[0], 'units'):
    #                u=lst[0].units
    #                return np.array([(a/u).magnitude for a in lst]) * u
    #            else:
    #                return np.array(lst)
    #        return ParamDB._ureg.Quantity(*args)



    def read(self, filename, regenerate_index=True):
        filename=os.path.join(self._folder,filename)
        with open(filename) as f:
            firstline=next(f)
            if firstline.strip()==("Omniscient v1"):
                return self._read_V1(f)
            else:
                raise Exception("Don't know how to parse this file."+\
                    "  First line should be something like 'Omniscient v1'")
        if regenerate_index: self.regenerate_index()

    def _read_V1(self,filehandle):
        """ Reads the parameter structure from the PyNitride parameters file into this database.

        :param filehandle: the opened filehandle to read from (first line "PyNitride ..." has already been read out)
        """

        # Chain of nested subdictionaries to populate, and their indices, as we read through the file
        chain=[]

        # The subdictionary declared on the previous line of the file, and its indentation
        prev_dict=self._dict
        prev_indent=-1

        # Go through valid lines without comments
        enumlines=enumerate([l.rstrip().split('#')[0] for l in filehandle])
        for i,line in enumlines:
            if line.strip()=="": continue

            # If this line is indented further than the previous, add the previous to the chain
            indent=len(line)-len(line.lstrip())
            if indent>prev_indent:
                # ... making sure the previous line actually declared a subdict (ie the previous was not a value line)
                if prev_dict is None:
                    raise Exception(
                        ("Syntax error: why is line {:d} of {:s} "+ \
                         "indented further than the previous line?").format(i,filehandle.name))
                chain+=[[prev_dict,prev_indent]]

            # If this line is indented less than the previous, remove the previous from the chain
            else:
                while indent<=chain[-1][1]:
                    chain.pop()

            # Now get rid of indents
            line=line.strip()

            # If this is a special interpreter line
            if line.startswith("<<<"):
                if "tabulated" in line:
                    self._detabulator(specialline=line,enumlines=enumlines,prev_dict=prev_dict)
                prev_dict=None

            # If this is a value line
            elif ":" in line and not line.endswith(":"):

                # Break at colon
                k,v=[p.strip() for p in line.split(":",maxsplit=1)]

                v=Brain.Value(v)

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
                        chain[-1][0][line]=OrderedDict()
                    prev_dict=chain[-1][0][line]
                # But if the key is '.', add as blank dict
                else:
                    prev_dict=OrderedDict()
                    chain[-1][0]+=[prev_dict]

            prev_indent=indent


    # Container for lazy evaluation of read-in values
    class Value():
        def __init__(self,valstr):
            self._valstr=valstr
        def get(self):
            if not hasattr(self,'_parsed'):
                valstr=self._valstr

                # if it's Python code, evaluate
                if valstr.startswith('`'):
                    self._parsed=eval(valstr[1:-1],np.__dict__)

                # if it's a string, return without quotes
                elif valstr.startswith('"') or valstr.startswith("'"):
                    self._parsed=valstr[1:-1]

                # if it's a boolean, return it
                elif valstr in ['True','False']:
                    return (valstr=="True")

                else:
                    # otherwise, pass through Pint
                    self._parsed=Brain._ureg(valstr)

            return self._parsed

    def _detabulator(self,specialline,enumlines,prev_dict):

        _,major,func=(specialline.strip()[3:]).strip().split(maxsplit=2)
        assert major=="row-major", "Not implemented non row-major tables"
        func=eval(func)

        i,line=next(enumlines)
        colnames=line.strip().split()
        ind=colnames[0]
        colnames=colnames[1:]

        for i,line in enumlines:
            if line.strip().startswith(">>>"):
                return
            vals=line.strip().split()
            vind=vals[0]
            vals=[Brain.Value(func(valstr,vind,c)) for valstr,c in zip(vals[1:],colnames)]
            prev_dict[ind+"="+vind]=OrderedDict(zip(colnames,vals))
Brain._ureg=UnitRegistry()

