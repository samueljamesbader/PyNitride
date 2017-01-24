import pytest
from poissolve.context import *
from poissolve.materials import read_1dp_mat, Material
from os.path import expanduser, join
import re

indir=expanduser("~/Jena/QWHEMT/Modeling/1dp/")

if __name__=="__main__": pass
    #pytest.main(args=[__file__])


# Current limitations.  Layer lines must contain t=(...)nm
# Dopants max one per line.
def import_1dp_input(fileprefix):
    read_1dp_mat()

    filename=join(indir,fileprefix+".txt")
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
                    doping+=[[dopetype,float(dopeconc)]]
                else:
                    doping+=[[]]

            mo=re.match("dy\s*=\s*([\d\.eE]+)",line)
            if mo: dz=float(mo.group(1))/10

            mo=re.match("schrodingerstart\s*=\s*([\d\.eE\-\+]+)",line)
            if mo: sstart=float(mo.group(1))
            mo=re.match("schrodingerstop\s*=\s*([\d\.eE\-\+]+)",line)
            if mo: sstop=float(mo.group(1))

    mesh=Mesh(EpiStack(*layers), dz, uniform=True)
    assert np.allclose(mesh._dz,dz,atol=1e-10,rtol=0), "Meshing failed."

    for dopetype in set(d[0] for d in doping if len(d)):
        mesh[{'Na':'Acceptor','Nd':'Donor','Naa':'DeepAcceptor','Ndd':'DeepDonor'}[dopetype]]=\
            RegionFunction(mesh,lambda li: doping[li][1] if len(doping[li]) and doping[li][0]==dopetype else 0,pos='point')

    P=MaterialFunction(mesh,'P')
    mesh['rho_pol']=P.differentiate(fill_value=0.0)

    if sstart is not None and sstop is not None:
        sm=mesh.submesh([sstart,sstop])
    else: sm=None

    return mesh,sm


def import_1dp_output(fileprefix,m,sm):
    filename=join(indir,fileprefix+"_Out.txt")
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

    filename=join(indir,fileprefix+"_Status.txt")
    v={}
    with open(filename) as f:
        for line in (l.strip() for l in f):
            mo=re.match("(.+) eigenvalue \d+ =\s+([\deE\+\-\.]+)\s+eV",line)
            if mo:
                b={'Electron':'e_Gamma', 'Heavy hole': 'h_HH', 'Light hole': 'h_LH', 'Split-off hole': 'h_SO'}[mo.groups()[0]]
                if b not in v: v[b]=[]
                v[b]+=[float(mo.groups()[1])]
    for b,vi in v.items(): sm['Energies_'+b]=ConstantFunction(sm,vi)



def nothing():
    mpl.figure()
    mpl.plot(x, Ec, linewidth=2)
    mpl.plot(x, Ev, linewidth=2)
    mpl.plot(x, EF, linewidth=2)
    mpl.ylim(-6, 6)
    mpl.ylabel('Energy [eV]')
    mpl.xlabel('Depth [nm]')
    mpl.gca().twinx()

    mpl.plot(x, n / 1e21)
    mpl.plot(x, p / 1e21)
    mpl.xlim(0, 50)
    mpl.ylabel('$n$, $p$ [$10^{21}\mathrm{cm}^{-3}$]')

    #print("sigma_n: {:.3g}".format(np.trapz(n * ((x > 5) & (x < 40)), x) / 1e7))
    # print("sigma_p: {:.3g}".format(np.trapz(p*((x<25) & (x>7)),x)/1e7))


if __name__=='__main__':
    m,sm=import_1dp_input("GaNQWHEMT")
    import_1dp_output("GaNQWHEMT",m,sm)
    plot_carrierFV(m)

if 0:
    plot_wavefunctions(sm,bands=['e_Gamma'])

    sam=sm['Ec'][sm.index(6.5)]
    print("sam {:.2g}".format(sam))

    f=mpl.gcf()


    mpl.figure()
    check=((-hbar**2/2*(sm['Psi_e_Gamma'].differentiate()/MaterialFunction(sm,['ladder','electron','Gamma','mzs'])).differentiate()+sm['Ec']*sm['Psi_e_Gamma'])/sm['Psi_e_Gamma'])
    mpl.plot(sm.z,check.T)
    mpl.ylim(-10,10)
    mpl.xlim(0,15)
    mpl.title("Snider")


    mpl.sca(f.get_axes()[0])
    #m['Psi_e_Gamma']
    #my_n=SchrodingerSolver.carrier_density(sm['Psi_e_Gamma'],2,Material('qGaN')['ladder','electron','Gamma','mxys'],(sm['EF']-sm['Energies_e_Gamma'])/kT)
    #my_p=SchrodingerSolver.carrier_density(sm['Psi_h_HH'],2,Material('qGaN')['ladder','hole','HH','mxys'],-(sm['EF']-sm['Energies_h_HH'])/kT)
    #mpl.plot(sm.z,my_n/(1/cm**3),'r--',linewidth=2)
    #mpl.plot(sm.z,my_p/(1/cm**3),'r--',linewidth=2)

    del sm._functions['Psi_e_Gamma']
    del sm._functions['Psi_h_HH']
    del sm._functions['Psi_h_LH']
    del sm._functions['Energies_e_Gamma']
    del sm._functions['Energies_h_HH']
    del sm._functions['Energies_h_LH']
    ss=SchrodingerSolver(sm)
    ss.solve()
    mpl.plot(sm.z,sm['n']/(1/cm**3),'r--',linewidth=2)
    mpl.plot(sm.z,sm['p']/(1/cm**3),'r--',linewidth=2)

    mpl.sca(mpl.gcf().get_axes()[-2])
    plot_wavefunctions(sm,bands=['e_Gamma'])

    sam=sm['Ec'][sm.index(6.5)]
    print("sam {:.2g}".format(sam))

    mpl.figure()
    check=((-hbar**2/2*(sm['Psi_e_Gamma'].differentiate()/MaterialFunction(sm,['ladder','electron','Gamma','mzs'])).differentiate()+sm['Ec']*sm['Psi_e_Gamma'])/sm['Psi_e_Gamma'])
    mpl.plot(sm.z,check.T)
    mpl.ylim(-10,10)
    mpl.xlim(0,15)


    mpl.title("Me")
