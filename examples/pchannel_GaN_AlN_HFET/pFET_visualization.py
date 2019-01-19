import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import numpy as np
from pynitride.paramdb import to_unit, hbar, m_e

def plot_cond(m,mbkp):
    """ Custom plot of the dispersion, band diagram, DOS, and effective masses."""
    rmesh = mbkp.rmesh
    energy = mbkp.kpen
    print("{:.3g} x 10^13/cm^2".format(to_unit(float(m.p.integrate(definite=True)), "1/cm^2") / 1e13))
    plt.figure(figsize=(8, 8))

    # Three joined subplots
    gs2 = GridSpec(10, 2, wspace=0, hspace=0, left=.2, right=.8)

    # E-k Dispersion
    axbs = plt.subplot(gs2[5:, 1])
    for i, c in zip(range(6), ['b', 'b', 'r', 'r', 'g', 'g']):
        absk1=rmesh.absk1
        plt.plot(absk1, mbkp.interp_energy(absk1,0)) * 1e3, c)
        plt.plot(absk1, mbkp.interp_energy(absk1,pi/2)) * 1e3, c)
    plt.xlim(-4, 4)
    plt.axvline(0, color='k')
    plt.axhline(0, color='k', linestyle='--')
    plt.setp(axbs.get_yticklabels(), visible=False)
    plt.xlabel(r"$\leftarrow k_y\quad [1/nm]\quad k_x \rightarrow$")
    axbs.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True, which='both')

    return

    # DOS(E)
    plt.subplot(gs2[5:, 0], sharey=axbs)
    _kx, _ky = np.linspace(kmesh.kx1[0], kmesh.kx1[-1], 1000), np.linspace(kmesh.ky1[0], kmesh.ky1[-1], 1000)
    d_kx = _kx[1] - _kx[0]
    d_ky = _ky[1] - _ky[0]
    for i, c in zip([0, 2], ['b', 'r']):
        E = np.ravel(sim._enbv[i](_kx, _ky, grid=True))
        hist, bin_e = np.histogram(E, bins=100, range=(np.min(E), np.max(E) + .01))
        DOS = hist * d_kx * d_ky / (4 * np.pi ** 2) / np.diff(bin_e)
        E = (bin_e[1:] + bin_e[:-1]) / 2

        # print(list(zip(1e3*E,NE/(dkx*dky/(4*np.pi**2)))))
        # DOS=savgol_filter(DOS,5,1)
        plt.plot(DOS, E * 1e3, c)
        plt.fill_betweenx(E * 1e3, DOS * 1 / (1 + np.exp(-E / .026)), color=c, alpha=1)
        print("plevel: ", np.trapz(DOS * 1 / (1 + np.exp(-E / .026)), E))
    plt.xlim(0, 6)
    plt.ylim(-100, 25)
    plt.yticks([25, 0, -25, -50, -75])
    plt.ylabel("Energy [meV]")
    plt.xticks([0, 2, 4])
    plt.xlabel("DOS [eV$^{-1}$nm$^{-2}$]      $\ $", )
    plt.axhline(0, color='k', linestyle='--')
    plt.annotate("Filled", (1.2, -30), xytext=(1.4, -50),
                 arrowprops=dict(connectionstyle='arc3,rad=.1', color='b'), color='b')
    plt.annotate("Total", (4.2, -80), xytext=(1.8, -65),
                 arrowprops=dict(connectionstyle='arc3,rad=.1', color='b'), color='b')
    # plt.text(.6,-80,"Total")
    # plt.text(.4,-30,"Full")
    plt.grid(True, which='both', axis='y')

    # Band diagram
    gs1 = GridSpec(1, 1, left=.22, right=.42, bottom=.55)
    axbd = plt.subplot(gs1[0, 0])
    plt.plot(sim._m.Ec, sim._m._zp, 'g')
    plt.plot(sim._m.Ev, sim._m._zp, 'b')
    plt.plot(sim._m.EF, sim._m._zp, 'k--')
    plt.ylim(15, 0)
    plt.ylabel("Depth [nm]")
    # plt.xlabel("Energy [eV]")
    plt.xticks([])
    plt.text(.65, .1, "$E_C$", transform=axbd.transAxes, color='g')
    plt.text(.01, .1, "$E_V$", transform=axbd.transAxes, color='b')
    plt.text(.38, .1, "$E_F$", transform=axbd.transAxes, color='k')
    plt.twiny()
    plt.fill_between(sim._m.p, sim._m._zp, alpha=.5, color='purple')
    plt.xlim(0, .6)
    plt.yticks([4, 8, 12])
    plt.xticks([])
    plt.text(np.max(sim._m.p) * .75, sim._m._zp[np.argmax(sim._m.p)] + 2, "$p$", color='purple')

    # Mass(E)
    plt.subplot(gs2[1:5,1],sharex=axbs)
    def km(xory,eig,linespec):
        k= kmesh.kx1p if xory=="x" else kmesh.ky1p
        en=kmesh.along(energy[:,eig],dir=xory,onesided=True)
        m=-hbar**2/m_e/(np.diff(np.diff(en))/((k[1]-k[0]))**2)
        m[(m<0)]=10000
        plt.plot((-1)**(xory=='y')*k[1:-1],m,linespec)
    km(xory='x',eig=0,linespec='b')
    km(xory='x',eig=2,linespec='r')
    km(xory='y',eig=0,linespec='b')
    km(xory='y',eig=2,linespec='r')
    plt.ylim(0,3)
    plt.yticks([.5,1,1.5,2,2.5])
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.axvline(0,color='k')
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.ylabel("Effective mass [$m_e$]")
    plt.grid(True,which='both',axis='x')

    # # Velocity (E)
    # plt.subplot(gs2[1:5, 1], sharex=axbs)

    # def km(xory, eig, linespec):
    #     k = kmesh.kx1p if xory == "x" else kmesh.ky1p
    #     en = kmesh.along(energy[:, eig], dir=xory, onesided=True)
    #     v = np.diff(en) / (k[1] - k[0]) / hbar
    #     plt.plot((-1) ** (xory == 'y') * (k[1:] + k[:-1]) / 2, to_unit(v, "1e7 cm/s"), linespec)

    # km(xory='x', eig=0, linespec='b')
    # km(xory='x', eig=2, linespec='r')
    # km(xory='y', eig=0, linespec='b')
    # km(xory='y', eig=2, linespec='r')
    # plt.ylim(0, -3)
    # # plt.yticks([.5,1,1.5,2,2.5])
    # plt.setp(plt.gca().get_xticklabels(), visible=False)
    # plt.axvline(0, color='k')
    # plt.gca().yaxis.tick_right()
    # plt.gca().yaxis.set_label_position("right")
    # plt.ylabel("Group velocity [x$10^7$cm/s]")
    # plt.grid(True, which='both', axis='x')

    # Fermi(k)
    plt.subplot(gs2[0, 1], sharex=axbs)
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.yticks(va='top')
    plt.ylim(0, 1)
    plt.axvline(0, color='k')
    for i, c in zip(range(6), ['b', 'b', 'r', 'r', 'g', 'g']):
        k = kmesh.kx1p
        en = kmesh.along(energy[:, i], 'x', onesided=True)
        plt.fill_between(k, 1 / (1 + np.exp(-en / .026)), color=c)
        k = kmesh.ky1p
        en = kmesh.along(energy[:, i], 'y', onesided=True)
        plt.fill_between(-k, 1 / (1 + np.exp(-en / .026)), color=c)
    plt.text(.98, .98, "$f(k_x)$", transform=plt.gca().transAxes, va='top', ha='right', color='k')
    plt.text(.02, .98, "$f(k_y)$", transform=plt.gca().transAxes, va='top', ha='left', color='k')
    plt.xlim(-4, 4)