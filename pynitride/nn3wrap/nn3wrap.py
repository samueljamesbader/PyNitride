import re
from ipywidgets import HTML
from collections import OrderedDict
import os
from pynitride.util.paths import cd
from pynitride import config
from subprocess import call
from configparser import ConfigParser
from hashlib import md5
from shutil import rmtree
import numpy as np

class NN3Input():
    def __init__(self, macrofile):
        self._macrofile=macrofile
        self._params=OrderedDict()
        with open(macrofile) as mf:
            self._lines=[l for l in mf]
            for i,line in enumerate(self._lines):
                mo=re.match(r"%(?P<param>[^\s=]+)\s*=\s*(?P<val>[^\!]+)(?P<rest>.*)",line.strip())
                if mo and ("DoNotShow" not in mo.groupdict()["rest"]):
                    self._params[mo.groupdict()["param"]]={'val':mo.groupdict()["val"],'comment':mo.groupdict()["rest"].split("!")[-1],'line':i}

    def paramtable(self):
        outtable=HTML()
        outtable.value="<style>td {padding: 10px}</style><table border='1'>"
        for param,p in self._params.items():
            outtable.value+="<tr><td>"+param+"</td><td>"+p['val']+"</td><td>"+p['comment']+"</td></tr>"
        outtable.value+="</table>"
        return outtable

    def run(self, force=False, tweaks={}):
        for k in tweaks.keys():
            if k not in self._params:
                raise Exception("Unrecognized parameter: "+k)

        if not os.path.exists("Output"):
            os.makedirs("Output")
        out=self._macrofile[:-3]+"__"
        h=md5()

        with open(out+".in",'w') as outfile:
            params=iter(self._params.items())
            param,p=next(params)
            for i,line in enumerate(self._lines):
                if i==p['line']:
                    if param in tweaks:
                        val=tweaks[param]
                    else:
                        val=p['val']
                    l="%"+param+" = " + val + " ! " + p['comment']+"\n"
                    try:
                        param,p=next(params)
                    except:
                        p={'line':-1}
                else:
                    l=line
                outfile.write(l)
                h.update(l.encode('utf-8'))

        infile=os.path.abspath(out+".in")
        outdir=os.path.abspath("Output/"+out+h.hexdigest())

        if os.path.isdir(outdir):
            if not force:
                os.remove(infile)
                return NN3Output(outdir)
            else:
                rmtree(outdir)

        with cd(config['nn3wrap']['NEXTNANO3']):
            call(["./nextnano3.exe","-inputfile",infile,"-outputdirectory",outdir],stdout=open('/dev/stdout', 'w'))
        os.remove(infile)
        return NN3Output(outdir)

class NN3Output():

    names={
        'bd':"band_structure/BandEdges1D.dat",
        'el':"densities/density1Del.dat",
        'hl':"densities/density1Dhl.dat",
        'piezo':"densities/density1Dpiezo.dat",
        'pyro':"densities/density1Dpyro.dat",
        'dop':"doping_profile1D.dat",
        'int_el':"densities/int_el_dens1D.dat",
        'int_hl':"densities/subband1D_hl6x6kp_qc1_integrated.dat",
    }

    def __init__(self,dir):
        self.dir=dir

    def __getattr__(self, item):
        if item in NN3Output.names:
            setattr(self,item,NN3Output.read(os.path.join(self.dir,NN3Output.names[item])))
            return getattr(self,item)
        else:
            raise Exception("Don't recognize name: "+item)

    @staticmethod
    def read(filename):

        # Open the file
        with open(filename) as f:

            # Column names from the first line
            colnames=[n.strip() for n in next(f).strip().split()]

            # Collect the data from all following lines
            data=[[float(v) for v in l.strip().split()] for l in f]

            # Return dictionary of column name -> data
            return dict(zip(colnames,([np.array(c) for c in zip(*data)])))
