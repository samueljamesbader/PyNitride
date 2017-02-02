import matplotlib.pyplot as mpl
from pynitride.paramdb import ParamDB
cm,MV,k,hbar,q,eV=ParamDB().quantity("cm,MV,k,hbar,e,eV")
import numpy as np

def plot_QFV(mesh):
    mpl.figure()
    ax1=mpl.subplot(311)
    mpl.plot(mesh.zp, (mesh['rho'] - mesh['rho_pol']) / (q / cm ** 3))
    mpl.ylabel("Free charge [$|e|/\\mathrm{cm}^3$]")
    mpl.ylim([-np.max(mpl.ylim()),np.max(mpl.ylim())])
    mpl.twinx()
    mpl.plot(mesh.zp, mesh['rho_pol'] * mesh._dzm / (q / cm ** 2), 'k')
    mpl.ylim([-np.max(mpl.ylim()),np.max(mpl.ylim())])
    mpl.ylabel("Polarization charge [$|e|/\\mathrm{cm}^2$]")
    mpl.subplot(312,sharex=ax1)
    mpl.plot(mesh.zm,mesh['E']/(MV/cm))
    mpl.ylabel("Field [$\\mathrm{MV}/\\mathrm{cm}$]")
    mpl.subplot(313,sharex=ax1)
    mpl.plot(mesh.zp, mesh['Ec'] / eV)
    mpl.plot(mesh.zp, mesh['Ev'] / eV)
    mpl.plot(mesh.zp, mesh['EF'] / eV)
    mpl.ylabel("Bands [$\\mathrm{eV}$]")
    mpl.show()

def plot_carrierFV(mesh):
    mpl.figure()
    ax1=mpl.subplot(311)
    mpl.plot(mesh.zp, mesh['n'] / (1 / cm ** 3), 'b')
    mpl.plot(mesh.zp, mesh['p'] / (1 / cm ** 3), 'g')
    mpl.ylabel("Free carrier [$1/\\mathrm{cm}^3$]")
    mpl.ylim([-np.max(mpl.ylim()),np.max(mpl.ylim())])
    mpl.twinx()
    mpl.plot(mesh.zp, mesh['rho_pol'] * mesh._dzm / (q / cm ** 2), 'k')
    mpl.ylim([-np.max(mpl.ylim()),np.max(mpl.ylim())])
    mpl.ylabel("Polarization charge [$|e|/\\mathrm{cm}^2$]")
    mpl.subplot(312,sharex=ax1)
    mpl.plot(mesh.zm,mesh['E']/(MV/cm))
    mpl.ylabel("Field [$\\mathrm{MV}/\\mathrm{cm}$]")
    mpl.subplot(313,sharex=ax1)
    mpl.plot(mesh.zp, mesh['Ec'] / eV)
    mpl.plot(mesh.zp, mesh['Ev'] / eV)
    mpl.plot(mesh.zp, mesh['EF'] / eV)
    mpl.ylabel("Bands [$\\mathrm{eV}$]")
    mpl.show()
#plot_QFV(pn._mesh)

def plot_wavefunctions(mesh,bands=['e_Gamma']):
    m=mesh
    z=mesh.zp
    #mpl.plot(z,m['Ec'],'-')
    #mpl.plot(z,m['Ec_eff'][0],'-')
    #mpl.plot(z,m['Ev'],'-')

    for b in bands:
        E=m['Energies'+'_'+b][:,0]
        wf=np.min(np.diff(E))/np.max(np.abs(m['Psi'+'_'+b]))
        for i,Ei in enumerate(E[:1]):
            l=mpl.plot(z,Ei+0*(z),'--')[0]
            mpl.plot(z,m['Psi'+'_'+b][i,:]*wf+Ei,color=l.get_color())

    mpl.xlabel('$z$ [nm]')
    mpl.ylabel('Energy [eV]')

    mpl.twinx()
    if b[0][0]=='e':
        mpl.plot(z,m['n'],'.-k')
    else:
        mpl.plot(z,m['p'],'.-k')
    #mpl.plot(z,m['nderiv']*kT,'.-r')
    #mpl.yscale('log')
    mpl.yticks([])

