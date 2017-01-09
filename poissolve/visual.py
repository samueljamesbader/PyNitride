import matplotlib.pyplot as mpl
from poissolve.constants import q,cm,MV,eV
import numpy as np

def plot_QFV(mesh):
    mpl.figure()
    ax1=mpl.subplot(311)
    mpl.plot(mesh.z,mesh['rho']/(q/cm**3))
    mpl.plot(mesh.z,(mesh['rho']-mesh['rho_pol'])/(q/cm**3),'.--')
    mpl.ylabel("Charge [$|e|/\\mathrm{cm}^3$]")
    mpl.subplot(312,sharex=ax1)
    mpl.plot(mesh.zp,mesh['E']/(MV/cm))
    mpl.ylabel("Field [$\\mathrm{MV}/\\mathrm{cm}$]")
    mpl.subplot(313,sharex=ax1)
    mpl.plot(mesh.z,mesh['Ec']/eV)
    mpl.plot(mesh.z,mesh['Ev']/eV)
    mpl.plot(mesh.z,mesh['EF']/eV)
    mpl.ylabel("Bands [$\\mathrm{eV}$]")
    mpl.show()
#plot_QFV(pn._mesh)

def plot_wavefunctions(mesh,bands=['e_Gamma']):
    m=mesh
    z=mesh.z
    mpl.plot(z,m['Ec'],'.')

    for b in bands:
        E=m['E_i'+'_'+b][:,0]
        wf=np.min(np.diff(E))/np.max(np.abs(m['Psi_i'+'_'+b]))
        for i,Ei in enumerate(E):
            l=mpl.plot(z,Ei+0*(z),'--')[0]
            mpl.plot(z,m['Psi_i'+'_'+b][i,:]*wf+Ei,color=l.get_color())

    mpl.xlabel('$z$ [nm]')
    mpl.ylabel('Energy [eV]')

    mpl.twinx()
    mpl.plot(z,m['n'],'.-k')
    mpl.yticks([])

