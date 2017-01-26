import pytest
from pynitride import *
from os.path import expanduser, join
import numpy as np
from pynitride.poissolve.visual import plot_carrierFV, plot_wavefunctions
from pynitride.external.snider import import_1dp_input, import_1dp_output
from pynitride.poissolve.solvers.coupled import Coupled_Schrodinger_Poisson, Coupled_FD_Poisson
from pynitride.poissolve.solvers.fermidirac import FermiDirac3D


if __name__=="__main__": pass
    #pytest.main(args=[__file__])

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

def doit():
    pdb=ParamDB()
    pdb.clear()
    pdb.read_file('/usr/local/bin/materials',from_root='False')

    indir=expanduser(join(ROOT_DIR,"tests","1DPoisson_Runs"))
    m,sm=import_1dp_input(join(indir,"GaN_AlN_HEMT"))
    #import_1dp_output(join(indir,"GaN_AlN_HEMT"),m,sm)
    #plot_carrierFV(m)

    ParamDB()['material','GaN','conditions','default','bands','barrier']['GenericMetal']=.6*eV

    i=0
    def callback():
        nonlocal i
        i+=1
        if i>400:
            plot_carrierFV(m)
            return 1
    #csp=Coupled_Schrodinger_Poisson(m,schrodinger=sm)
    #csp.solve(callback=callback,low_act=3,rise=40)
    csp=Coupled_FD_Poisson(m)
    csp.solve(callback=callback,rise=40)

    return m,sm,csp




if __name__=='__main__':
    import cProfile
    #cProfile.run("doit()",'crestats.txt')
    m,sm,csp=doit()
    import matplotlib.pyplot as mpl
    mpl.interactive(True)
    plot_carrierFV(m)
    #plot_wavefunctions(sm,bands=['e_Gamma'])

    #mpl.gca().title="Mine"
    if 1:
        indir=expanduser(join(ROOT_DIR,"tests","1DPoisson_Runs"))
        m2,sm2=import_1dp_input(join(indir,"GaN_AlN_HEMT"))
        import_1dp_output(join(indir,"GaN_AlN_HEMT"),m2,sm2)
        plot_carrierFV(m2)
        #plot_wavefunctions(sm,bands=['e_Gamma'])


        s_Ev=m2['Ev'].copy()
        s_rho=m2['rho'].copy()
        s_p=m2['p'].copy()
        s_n=m2['n'].copy()
        s_ndpmnam=m2['Ndp-Nam'].copy()

        rc=(m2['E']*MaterialFunction(m2,['dielectric','eps'])).differentiate()
        mpl.sca(mpl.gcf().get_axes()[0])
        to_unit(-rc,'cm**-3').plot('--')


        rc.plot('--')
        fd2=FermiDirac3D(m2)
        fd2.solve()
        to_unit(-m2['rho'],'cm**-3').plot('x')
        #mpl.ylim(1e9,1e23)
        #mpl.yscale('log')
        #mpl.interactive(False)

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
