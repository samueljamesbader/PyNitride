import pytest
from poissolve.context import *
from poissolve.materials import read_1dp_mat, Material
from os.path import expanduser, join
import re

indir=expanduser("~/1DPoisson/1D Poisson Beta 8g Linux Distribution/Input_Files_Examples/")

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
    x, Ec, Ev, F, EF, n, p, NdmNa = [np.array(a) for a in [x, Ec, Ev, F, EF, n, p, NdmNa]]
    m['Ec'],m['Ev'],m['EF'],m['n'],m['p'] = [PointFunction(m,a) for a in [Ec,Ev,EF,n,p]]
    #(m['E']-MaterialFunction(m,'DEc',pos='point'))
    m['E']=m['Ec'].differentiate()
    m['rho']=PointFunction(m,np.NaN)

    assert np.allclose(m.z,x,atol=1e-10,rtol=0), "Meshes don't match"
    #assert np.max(np.abs(np.diff(dx)))<1e-10, "Can only import uniform meshes from 1DPoisson"



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
    m,sm=import_1dp_input("GaNDelta")
    import_1dp_output("GaNDelta",m,sm)
    plot_carrierFV(m)
    mpl.sca(mpl.gcf().get_axes()[0])

