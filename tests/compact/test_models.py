from pynitride.compact.models import GaNHEMT_iMVSG, VO2Res, Direction, HyperFET
import timeit
import numpy as np
import matplotlib.pyplot as plt
import pytest
import cProfile
import warnings

if __name__=="__main__":
    pass
    #pytest.main(args=[__file__,"-s","--plots"])


def test_ganhemt():
    hemt=GaNHEMT_iMVSG()

    VD=np.array(3.0)
    VG=np.linspace(-7,2)
    I=hemt.ID(VD=VD,VG=VG)
    plt.figure()
    plt.plot(VG,I)
    plt.show()
    plt.gcf().canvas.set_window_title("HEMT")
    plt.yscale('log')

def test_VO2_Res():
    vo2=VO2Res(R_met=1e-3)

    trans=vo2.I_IMT
    I=np.linspace(0,4*trans)
    plt.figure()
    plt.plot(vo2.V(I,Direction.FORWARD),I)
    plt.plot(vo2.V(I,Direction.BACKWARD),I)
    plt.show()

def test_HyperFET():
    hemt=GaNHEMT_iMVSG()
    vo2=VO2Res(R_met=1e-3)
    hf=HyperFET(hemt,vo2)

    VD=np.array(3.0)
    VG=np.linspace(-5,-2,500)

    If,Ib=hf.I_double(VD=VD,VG=VG)
    #assert not(np.any(np.isnan(If)) or np.any(np.isnan(Ib))), "NaNs!"

    plt.figure()
    plt.plot(VG,If)
    plt.plot(VG,Ib)
    plt.show()
    plt.gcf().canvas.set_window_title("HyperFET")

    plt.yscale('log')




def test_HyperFET_approx():

    from pint import UnitRegistry
    ureg=UnitRegistry()
    si=lambda x: ureg(x).to_base_units().magnitude

    # Shukla given parameters
    # Table
    rho_m=si("5e-4 ohm cm")
    rho_i=si("80 ohm cm")
    J_MIT=si("2e6 A/cm^2") # SAMMMM
    J_IMT=si(".55e4 A/cm^2")

    # Text
    thickness=si("14nm")
    vo2W=si("14nm")
    vo2L=si("8nm")

    def VO2(W,L):
        I_IMT=J_IMT*thickness*W
        I_MIT=J_MIT*thickness*W

        R_ins=rho_i*L/(W*thickness)
        R_met=rho_m*L/(W*thickness)

        V_IMT=I_IMT*R_ins
        V_MIT=I_MIT*R_met

        return VO2Res(I_IMT=I_IMT, V_IMT=V_IMT, I_MIT=I_MIT, V_MIT=V_MIT, R_met=R_met)

    VT0=.35
    W=70
    Cinv_vxo=2500
    SS=.07
    alpha=0
    beta=1.8
    VDD=.5
    VDsats=.1
    delta=.2
    log10Gleak=-12

    plt.figure(figsize=(12,6))
    hemt=GaNHEMT_iMVSG(
        W=W*1e-9,Cinv_vxo=Cinv_vxo,
        VT0=VT0,alpha=alpha,SS=SS,delta=delta,
        VDsats=VDsats,beta=beta,eta=0,Gleak=10**log10Gleak)

    vo2=VO2(vo2W,vo2L)
    hf=HyperFET(hemt,vo2,VDD)
    #print(HyperFET.approx_shift(hemt,pcr))
    hemt2=hemt.shifted(hf.approx_shift())
    hf2=HyperFET(hemt2,vo2,VDD)

    VD=np.array(VDD)
    VG=np.linspace(0,.5,500)

    plt.subplot(131)
    I=hemt.ID(VD=VD,VG=VG)
    plt.plot(VG,I/hemt.W)


    If,Ib=[np.ravel(i) for i in hf.I_double(VD=VD,VG=VG)]
    plt.plot(VG[~np.isnan(If)],If[~np.isnan(If)]/hemt.W)
    plt.plot(VG[~np.isnan(Ib)],Ib[~np.isnan(Ib)]/hemt.W)

    plt.plot(hf.approx_hyst("Vleft"),vo2.I_MIT/hemt.W,'o')
    plt.plot(hf.approx_hyst("Vright"),vo2.I_IMT/hemt.W,'o')

    Ifa=hf.approx_I(VD=VD,VG=VG,region="lowernoleak")
    plt.plot(VG,Ifa/hemt.W,'--')
    Ifa=hf.approx_I(VD=VD,VG=VG,region="inversion")
    plt.plot(VG,Ifa/hemt.W,'--')
    Ifa=hf.approx_I(VD=VD,VG=VG,region="uppersub")
    plt.plot(VG,Ifa/hemt.W,'--')


    floor=10**log10Gleak*VD

    # Because yscale log complains about NaNs
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        plt.yscale('log')
        plt.ylim(1e-2,1e3)
        plt.xlabel("$V_{GS}\;\mathrm{[V]}$")
        plt.ylabel("$I/W\;\mathrm{[mA/mm]}$")



    plt.subplot(132)
    I=hemt.ID(VD=VD,VG=VG)
    plt.plot(VG,I/hemt2.W)
    If,Ib=[np.ravel(i) for i in hf2.I_double(VD=VD,VG=VG)]
    plt.plot(VG[~np.isnan(If)],If[~np.isnan(If)]/hemt2.W)
    plt.plot(VG[~np.isnan(Ib)],Ib[~np.isnan(Ib)]/hemt2.W)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        plt.yscale('log')
        plt.ylim(1e-2,1e3)
        #plt.xlabel("$V_{GS}\;\mathrm{[V]}$")
        #plt.ylabel("$I/W\;\mathrm{[mA/mm]}$")
        plt.yticks([])

    plt.subplot(133)
    I=hemt.ID(VD=VD,VG=VG)
    plt.plot(VG,I/hemt2.W)
    If,Ib=[np.ravel(i) for i in hf2.I_double(VD=VD,VG=VG)]
    plt.plot(VG[~np.isnan(If)],If[~np.isnan(If)]/hemt2.W)
    plt.plot(VG[~np.isnan(Ib)],Ib[~np.isnan(Ib)]/hemt2.W)

    plt.tight_layout()





def test_HyperFET_shift():
    VT0=.35

    delta=.00
    VD=5.
    vo2=VO2Res(R_met=1e-8, V_MIT=.05, I_IMT=.005e-3)
    hemt=GaNHEMT_iMVSG(VT0=VT0, alpha=0, delta=delta, eta=0, Gleak=0)
    shift=HyperFET(hemt,vo2,VDD=1.0).approx_shift()
    print("shift ",shift)
    hemtshifted=GaNHEMT_iMVSG(VT0=(VT0+shift), alpha=0, delta=delta, eta=0, Gleak=0)
    hf=HyperFET(hemtshifted,vo2,VDD=1.0)

    from numpy import exp
    Ioff=hemt.n*hemt.W*hemt.Cinv_vxo*hemt._Vth*exp(-hemt.VT0/(hemt.n*hemt._Vth))

    VD=np.array(VD)
    VG=np.linspace(0,VT0+5,500)

    If,Ib=hf.I_double(VD=VD,VG=VG)
    Ifa=hf.approx_I(VD=VD,VG=VG,region="lowernoleak")
    #assert not(np.any(np.isnan(If)) or np.any(np.isnan(Ib))), "NaNs!"

    plt.figure()
    plt.plot(VG,hemt.ID(VD,VG))
    plt.plot(VG,If)
    plt.plot(VG,Ib)
    plt.plot(VG,Ioff+0*VG)
    plt.plot(VG,Ifa,'--')
    plt.show()
    plt.gcf().canvas.set_window_title("HyperFET Shift Test")

    plt.yscale('log')

def test_weird_HyperFET():
    VT0=-5
    Cinv_vxo=70
    alpha=0
    beta=1.8
    VDsats=3.5
    log10Gleak=-12
    I_IMT=2.2e-3
    V_IMT=1.7
    I_MIT=1.2e-3
    V_MIT=.5
    R_met=0.01

    #plt.figure(figsize=(9,6))
    hemt=GaNHEMT_iMVSG(
        W=200e-6,Cinv_vxo=Cinv_vxo,
        VT0=VT0,alpha=alpha,SS=80e-3,delta=0,
        VDsats=VDsats,beta=beta,eta=0,Gleak=10**log10Gleak)
    vo2=VO2Res(I_IMT=I_IMT,V_IMT=V_IMT,I_MIT=I_MIT,V_MIT=V_MIT,R_met=R_met)

    hf=HyperFET(hemt,vo2)

    VD=np.array(50.0)
    VG=np.linspace(-7.5,0,500)

    try:
        If,Ib=[np.ravel(i) for i in hf.I_double(VD=VD,VG=VG)]
    except Exception as e:
        return
    plt.figure(figsize=(9,6))
    plt.plot(VG[~np.isnan(If)],If[~np.isnan(If)]/hemt.W)
    plt.plot(VG[~np.isnan(Ib)],Ib[~np.isnan(Ib)]/hemt.W)

    plt.plot(hf.approx_hyst("Vleft"),vo2.I_MIT/hemt.W,'o')
    plt.plot(hf.approx_hyst("Vright"),vo2.I_IMT/hemt.W,'o')

    floor=10**log10Gleak*VD

    plt.yscale('log')
    plt.ylim(floor,1e4)
    plt.xlabel("$V_{GS}\;\mathrm{[V]}$")
    plt.ylabel("$I/W\;\mathrm{[mA/mm]}$")


def test_HyperFET_timing():
    hemt=GaNHEMT_iMVSG()
    vo2=VO2Res(R_met=1e-3)
    hf=HyperFET(hemt,vo2)

    numvalues=500
    Nrepeat=10
    VD=np.array(3.0)
    VG=np.linspace(-5,2,numvalues)

    hf_timing=timeit.timeit("hf.I_double(VD=VD,VG=VG)",number=Nrepeat,globals=locals())/Nrepeat
    print("\nHyperFET computed {:d} values in {:.3e} s (average of {:d} runs)." \
          .format(numvalues,hf_timing,Nrepeat))

if __name__=="__main__":
    test_HyperFET_approx()
    #pytest.main(args=[__file__,"-s","--plots"])
