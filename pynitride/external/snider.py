from pynitride import *

import numpy as np
import re

# Current limitations.  Layer lines must contain t=(...)nm
# Dopants max one per line.
def import_1dp_input(fileprefix):
    filename=fileprefix+".txt"
    layers=[]
    doping=[]
    sstart=sstop=None
    with open(filename) as f:
        for line in (l.strip() for l in f):
            mo=re.match("^\s?(\w+)\s+.*t\s*=\s*([\d\.eE\-\+]+)\s*nm",line)
            if mo:
                matname,thickness=mo.groups()
                #if matname=="AlGaN": matname="qAlN"
                layers+=[[len(layers),matname,float(thickness)]]

                mo=re.match(".*(Na|Nd|Naa|Ndd)\s*=\s*([\d\.eE\-\+]+)",line)
                if mo:
                    dopetype,dopeconc=mo.groups()
                    doping+=[[dopetype,float(dopeconc)*cm**-3]]
                else:
                    doping+=[[]]

            mo=re.match("dy\s*=\s*([\d\.eE]+)",line)
            if mo: dz=float(mo.group(1))/10

            mo=re.match("schrodingerstart\s*=\s*([\d\.eE\-\+]+)",line)
            if mo: sstart=float(mo.group(1))
            mo=re.match("schrodingerstop\s*=\s*([\d\.eE\-\+]+)",line)
            if mo: sstop=float(mo.group(1))

    mesh=Mesh(EpiStack(*layers,surface='GenericMetal'), dz, uniform=True)
    assert np.allclose(mesh._dz,dz,atol=1e-10,rtol=0), "Meshing failed."

    for dopetype in set(d[0] for d in doping if len(d)):
        mesh[{'Na':'AcceptorActiveConc','Nd':'DonorActiveConc','Naa':'DeepAcceptorActiveConc','Ndd':'DeepDonorActiveConc'}[dopetype]]= \
            RegionFunction(mesh,lambda li: doping[li][1] if len(doping[li]) and doping[li][0]==dopetype else 0,pos='point')

    P=MaterialFunction(mesh,['polarization','Ptot'])
    mesh['rho_pol']=P.differentiate(fill_value=0.0)

    if sstart is not None and sstop is not None:
        sm=mesh.submesh([sstart,sstop])
    else: sm=None

    return mesh,sm


def import_1dp_output(fileprefix,m,sm):
    filename=fileprefix+"_Out.txt"
    with open(filename) as f:
        next(f)
        x = []
        Ec = []
        Ev = []
        F = []
        EF = []
        n = []
        p = []
        NdmNa = []
        for l in f:
            xi, Eci, Evi, Fi, EFi, ni, pi, NdmNai = [float(a) for a in l.strip().split()[:8]]
            x += [xi / 10]
            Ec += [Eci]
            Ev += [Evi]
            F += [Fi / 1e8]
            EF += [EFi]
            n += [ni]
            p += [pi]
            NdmNa += [NdmNai]
    x, Ec, Ev, F, EF= [np.array(a) for a in [x, Ec, Ev, F, EF]]
    n, p, NdmNa =[np.array(a)*(1/cm**3) for a in [n, p, NdmNa]]
    m['Ec'],m['Ev'],m['EF'],m['n'],m['p'] = [PointFunction(m,a) for a in [Ec,Ev,EF,n,p]]
    #(m['E']-MaterialFunction(m,'DEc',pos='point'))
    m['E']=m['Ec'].differentiate()
    m['rho']=PointFunction(m,np.NaN)

    assert np.allclose(m.z,x,atol=1e-10,rtol=0), "Meshes don't match"


    #if not sm: return
    #filename=join(indir,fileprefix+"_Wave.txt")
    #with open(filename) as f:
    #headers=next(f)
    #subbands=headers.split()[2::3]
    #
    #vals=[]
    #for line in (l.strip() for l in f):
    #vals+=[[float(x) for x in line.split()]]
    #vals=np.array(vals)*np.sqrt(10) # Angstrom to nm
    #
    #for b in set(subbands):
    #indices=[i+1 for i,bi in enumerate(subbands) if bi==b]
    #sm['Psi_'+{'el':'e_Gamma','hh':'h_HH','lh':'h_LH'}[b]]=\
    #PointFunction(sm,vals[:,min(indices):max(indices)+1].T)

    filename=fileprefix+"_Status.txt"
    v={}
    with open(filename) as f:
        for line in (l.strip() for l in f):
            mo=re.match("(.+) eigenvalue \d+ =\s+([\deE\+\-\.]+)\s+eV",line)
            if mo:
                b={'Electron':'e_Gamma', 'Heavy hole': 'h_HH', 'Light hole': 'h_LH', 'Split-off hole': 'h_SO'}[mo.groups()[0]]
                if b not in v: v[b]=[]
                v[b]+=[float(mo.groups()[1])]
    for b,vi in v.items(): sm['Energies_'+b]=ConstantFunction(sm,vi)

