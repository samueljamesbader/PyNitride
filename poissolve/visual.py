import matplotlib.pyplot as mpl
from poissolve.constants import q,cm,MV,eV

def plot_QFV(mesh):
    mpl.figure()
    ax1=mpl.subplot(311)
    mpl.plot(mesh.z,mesh['rho']/(q/cm**3))
    mpl.plot(mesh.z,(mesh['rho']-mesh['rho_pol'])/(q/cm**3),'--')
    mpl.ylabel("Charge [$|e|/\\mathrm{cm}^3$]")
    mpl.subplot(312,sharex=ax1)
    mpl.plot(mesh.zp,mesh['E']/(MV/cm))
    mpl.ylabel("Field [$\\mathrm{MV}/\\mathrm{cm}$]")
    mpl.subplot(313,sharex=ax1)
    print(mesh['Ec'])
    mpl.plot(mesh.z,mesh['Ec']/eV)
    mpl.plot(mesh.z,mesh['Ev']/eV)
    mpl.plot(mesh.z,mesh['EF']/eV)
    mpl.ylabel("Bands [$\\mathrm{eV}$]")
    mpl.show()
#plot_QFV(pn._mesh)