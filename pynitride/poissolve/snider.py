from pynitride import ParamDB, EpiStack, Mesh, PointFunction, ROOT_DIR, ConstantFunction, RegionFunction, MaterialFunction
pmdb=ParamDB(units='neu')
q,cm=pmdb.quantity("e,cm")

from textwrap import dedent
import numpy as np
import re
import os.path

import time
import datetime


# Current limitations.  Layer lines must contain t=(...)nm
# Dopants max one per line.
def import_1dp_input(fileprefix,pmdb=ParamDB()):
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

    mesh=Mesh(EpiStack(*layers,surface='GenericMetal',pmdb=pmdb), dz, uniform=True)
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
    m['Ec'],m['Ev'],m['EF'],m['n'],m['p'],m['Ndp-Nam'] = [PointFunction(m,a) for a in [Ec,Ev,EF,n,p,NdmNa]]
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

def convert_1dpmat_to_PyNitride(matfilename,outfilename,to_root=True):
    T=300
    if to_root: outfilename=os.path.join(ROOT_DIR,"parameters",outfilename)
    with open(outfilename,'w') as out:
        ts=time.time()
        dt=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        out.write("PyNitride v2\n\n# Autoconverted from 1D Poisson materials.txt "+dt+"\n")
        out.write("#Assuming T=300K")
        with open(matfilename) as f:
            next(f)
            for line in f:
                mo=re.match(r"^(\w+)\s+binary\s+\w+",line.strip())
                if mo:
                    matname=mo.groups(0)[0]

                    tmp={}
                    next(f) # skip mystery zeros line in materials file
                    for line in f:
                        mo=re.match(r"(\w+)=([\d\*eE\+\-\.Temp\/\^\(\)]+)",line)
                        if mo is None: break
                        try:
                            tmp[mo.groups()[0]]=eval("(lambda Temp: "+mo.groups()[1].replace('^','**')+")("+str(T)+")")
                            #if matname=="GaN" or matname=="AlN":
                                #if mo.groups()[0]=='pol':
                                #print(mo.groups(),tmp[mo.groups()[0]])
                        except Exception as e:
                            print(e)
                            print(mo.groups())
                            import numpy as np
                            tmp[mo.groups()[0]]=np.NaN

                    print(tmp)
                    string=dedent("""
                    material={:s}
                        conditions=
                            Eg:{:.10g} eV
                            carrier=electron
                                DEc: {:.10g} eV
                                band=
                                    g: 2 * {:d} # including spin
                                    mzs:  {:.10g} m_e
                                    mxys: {:.10g} m_e
                                    mdos: {:.10g} m_e
                                    DE: 0 eV,
                            carrier=hole
                                band=HH
                                    g: 2 # including spin
                                    mzs:  {:.10g} m_e
                                    mxys: {:.10g} m_e
                                    mdos: {:.10g} m_e
                                    DE: 0 eV,
                                band=LH
                                    g: 2 # including spin
                                    mzs:  {:.10g} m_e
                                    mxys: {:.10g} m_e
                                    mdos: {:.10g} m_e
                                    DE: 0 eV,
                                band=SO
                                    g: 2 # including spin
                                    mzs:  {:.10g} m_e
                                    mxys: {:.10g} m_e
                                    mdos: {:.10g} m_e
                                    DE: 0 eV,
                        dielectric
                            eps: {:.10g} epsilon_0
                        dopant=Donor
                            type: 'Donor'
                            E: {:.10g} eV
                            g: 2
                        dopant=Acceptor
                            type: 'Acceptor'
                            E: {:.10g} eV
                            g: 4
                        dopant=DeepDonor
                            type: 'Donor'
                            E: {:.10g} eV
                            g: 2
                        dopant=DeepAcceptor
                            type: 'Acceptor'
                            E: {:.10g} eV
                            g: 4
                        polarization
                            Ptot: {:.10g} C/cm**2
                    """.format(matname,tmp['eg'],tmp['dec'],
                               int(tmp['val']),tmp['me'],tmp['me'],tmp['me'],
                               tmp['mh'],tmp['mh'],tmp['mh'],
                               tmp['mlh'],tmp['mlh'],tmp['mlh'],
                               tmp['mhso'],tmp['mhso'],tmp['mhso'],
                               tmp['er'],
                               tmp['ed'],tmp['ea'],tmp['edd'],tmp['eda'],
                               -tmp['pol']))
                    out.write(string)
                #import scipy.constants as const
                #self['material',matname,'conditions','default']=dict(
                #    bands=dict(
                #        Eg=tmp['eg'] *eV,
                #        DEc=tmp['dec'] *eV,
                #        barrier=dict(),
                #        electron=dict(
                #            Gamma=dict(g=2*tmp['val'],mzs=tmp['me']*m_e,mxys=tmp['me']*m_e,mdos=tmp['me']*m_e,DE=0)),
                #        hole=dict(
                #            HH=dict(g=2,mzs=tmp['mh']*m_e,mxys=tmp['mh']*m_e,mdos=tmp['mh']*m_e,DE=0),
                #            LH=dict(g=2,mzs=tmp['mlh']*m_e,mxys=tmp['mlh']*m_e,mdos=tmp['mlh']*m_e,DE=0),
                #            SO=dict(g=2,mzs=tmp['mhso']*m_e,mxys=tmp['mhso']*m_e,mdos=tmp['mhso']*m_e,DE=0))),
                #    dielectric=dict(eps=tmp['er']*epsilon_0),dopant=dict(
                #        Donor=dict(type='Donor',E=tmp['ed']*eV, g=2),
                #        Acceptor=dict(type='Acceptor',E=tmp['ea']*eV, g=4),
                #        DeepDonor=dict(type='Donor',E=tmp['edd']*eV, g=2),
                #        DeepAcceptor=dict(type='Acceptor',E=tmp['eda']*eV, g=4)),
                #    polarization=dict(Ptot=-tmp['pol']/const.elementary_charge/cm**2))

if __name__=="__main__":
    convert_1dpmat_to_PyNitride('/usr/local/bin/materials.txt',"1dp.txt")
