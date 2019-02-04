import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import numpy as np
from pynitride.paramdb import to_unit, hbar, m_e
pi=np.pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import numpy as np
from pynitride.paramdb import to_unit, hbar, m_e, kb, nm
from pynitride.maths import cart2polar
pi=np.pi

def conduction_band_panels(m,mbkp):
    """ Custom plot of the dispersion, band diagram, DOS, and effective masses."""
    rmesh = mbkp.rmesh
    energy = mbkp.kpen
    #print("{:.3g} x 10^13/cm^2".format(to_unit(float(m.p.integrate(definite=True)), "1/cm^2") / 1e13))
    plt.figure(figsize=(8, 8))

    # Three joined subplots
    gs2 = GridSpec(10, 2, wspace=0, hspace=0, left=.2, right=.8)

    # E-k Dispersion
    axbs = plt.subplot(gs2[5:, 1])
    for i, c in zip(range(6), ['b', 'b', 'r', 'r', 'g', 'g']):
        absk1=rmesh.absk1
        plt.plot(absk1, mbkp.interp_energy(absk1,0,eig=i) * 1e3, c)
        plt.plot(-absk1, mbkp.interp_energy(absk1,pi/2,eig=i) * 1e3, c)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim(np.min(mbkp.kpen[:,0])*1e3, np.max(mbkp.kpen[:,0])*1e3+5)
    plt.axvline(0, color='k')
    plt.axhline(0, color='k', linestyle='--')
    plt.setp(axbs.get_yticklabels(), visible=False)
    plt.xlabel(r"$\leftarrow k_y\quad [1/nm]\quad k_x \rightarrow$")
    #axbs.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True, which='both')


    # DOS(E)
    plt.subplot(gs2[5:, 0], sharey=axbs)
    kT=kb*m.T[0]
    (kx,dkx), (ky,dky) = [np.linspace(-rmesh.kmax, rmesh.kmax, 4000,retstep=True)]*2
    KX,KY=np.meshgrid(kx,ky)
    ABSK,THETA=cart2polar(KX,KY)
    absk,theta=ABSK[ABSK<=rmesh.kmax],THETA[ABSK<=rmesh.kmax]
    for i, c in zip([0, 2, 4], ['b', 'r', 'g']):
        E = mbkp.interp_energy(absk,theta,eig=i)
        hist, bin_e = np.histogram(E, bins=200, range=(np.min(E)-.01, np.max(E) ))
        DOS = hist * dkx * dky / (4 * np.pi ** 2) / np.diff(bin_e)
        E = (bin_e[1:] + bin_e[:-1]) / 2
        plt.plot(DOS, E * 1e3, c)
        plt.fill_betweenx(E * 1e3, DOS * 1 / (1 + np.exp( E / kT)), color=c, alpha=1)
    plt.xlim(0, .5)
    #plt.yticks([25, 0, -25, -50, -75])
    plt.ylabel("Energy [meV]")
    plt.xticks(plt.xticks()[0][:-1])
    plt.xlabel("DOS [eV$^{-1}$nm$^{-2}$]      $\ $", )
    plt.axhline(0, color='k', linestyle='--')
    #plt.annotate("Filled", (1.2, -30), xytext=(1.4, -50),
    #             arrowprops=dict(connectionstyle='arc3,rad=.1', color='b'), color='b')
    #plt.annotate("Total", (4.2, -80), xytext=(1.8, -65),
    #             arrowprops=dict(connectionstyle='arc3,rad=.1', color='b'), color='b')
    # plt.text(.6,-80,"Total")
    # plt.text(.4,-30,"Full")
    plt.grid(True, which='both', axis='y')

    # Band diagram
    gs1 = GridSpec(1, 1, left=.22, right=.42, bottom=.55)
    axbd = plt.subplot(gs1[0, 0])
    plt.plot(m.Ec, m.zm, 'g')
    plt.plot(m.Ev, m.zm, 'b')
    plt.plot(m.EF, m.zp, 'k--')
    plt.ylim(m._layers[0].thickness+5*nm, 0)
    plt.ylabel("Depth [nm]")
    plt.xticks([])
    plt.text(.65, .1, "$E_C$", transform=axbd.transAxes, color='g')
    plt.text(.01, .1, "$E_V$", transform=axbd.transAxes, color='b')
    plt.text(.38, .1, "$E_F$", transform=axbd.transAxes, color='k')
    plt.twiny()
    plt.fill_between(m.n, m.zp, alpha=.5, color='purple')
    plt.xlim(0, .6)
    plt.xticks([])
    plt.text(np.max(m.p) * .75, m.zp[np.argmax(m.p)] + 2, "$p$", color='purple')

    # Mass(E)
    plt.subplot(gs2[1:5,1],sharex=axbs)
    absk=np.linspace(0,rmesh.kmax,100)
    for i, c in zip([0, 2, 4], ['b', 'r', 'g']):
        ms=mbkp.interp_radial_eff_mass(absk,theta=0,eig=i)
        ms[(ms/m_e<0)|(ms/m_e>10)]=np.NaN
        plt.plot(absk,ms/m_e,c)

        ms=mbkp.interp_radial_eff_mass(absk,theta=pi/2,eig=i)
        ms[(ms/m_e<0)|(ms/m_e>10)]=np.NaN
        plt.plot(-absk,ms/m_e,c)

    plt.ylim(0,.4)
    #plt.yticks([.5,1,1.5,2,2.5])
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.axvline(0,color='k')
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.ylabel("Effective mass [$m_e$]")
    plt.grid(True,which='both',axis='x')

    # Fermi(k)
    plt.subplot(gs2[0, 1], sharex=axbs)
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.yticks(va='top')
    plt.ylim(0, 1)
    plt.axvline(0, color='k')
    absk=np.linspace(0,rmesh.kmax,100)
    for i, c in zip(range(6), ['b', 'b', 'r', 'r', 'g', 'g']):
        en = mbkp.interp_energy(absk,theta=0,eig=i)
        plt.fill_between( absk, 1 / (1 + np.exp(en / kT)), color=c)

        en = mbkp.interp_energy(absk,theta=pi/2,eig=i)
        plt.fill_between(-absk, 1 / (1 + np.exp(en / kT)), color=c)

    plt.text(.98, .98, "$f(k_x)$", transform=plt.gca().transAxes, va='top', ha='right', color='k')
    plt.text(.02, .98, "$f(k_y)$", transform=plt.gca().transAxes, va='top', ha='left', color='k')
    plt.autoscale(enable=True, axis='x', tight=True)
