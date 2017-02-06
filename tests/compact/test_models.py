from pynitride.compact.models import GaNHEMT_iMVGS, VO2Res, Direction, HyperFET
import timeit
import numpy as np
import matplotlib.pyplot as plt
import pytest
import cProfile

if __name__=="__main__":
    #pytest.main(args=[__file__,"-s","--plots"])
    pass


def test_ganhemt():
    hemt=GaNHEMT_iMVGS()

    VD=np.array(3.0)
    VG=np.linspace(-7,2)
    I=hemt.ID(VD=VD,VG=VG)
    #Ff=[hemt.Ff(VDS=VD,VGS=vg) for vg in VG]
    plt.figure()
    plt.plot(VG,I)
    #plt.plot(VG,Ff)
    plt.show()
    plt.gcf().canvas.set_window_title("HEMT")
    plt.yscale('log')

    #print(I)
    #print(Ff)


def test_VO2_Res():
    vo2=VO2Res(R_met=1e-3)

    trans=vo2.I_IMT

    I=np.linspace(0,4*trans)
    plt.figure()
    plt.plot(vo2.V(I,Direction.FORWARD),I)
    plt.plot(vo2.V(I,Direction.BACKWARD),I)
    plt.show()

    #I=np.linspace(fi_min,fi_max*4)
    #V=vo2.V(I)
    #plt.figure()
    #plt.plot(V,I)
    #plt.show()

def test_HyperFET():
    hemt=GaNHEMT_iMVGS()
    vo2=VO2Res(R_met=1e-3)
    hf=HyperFET(hemt,vo2)

    VD=np.array(3.0)
    VG=np.linspace(-5,-2,500)

    If,Ib=hf.I_double(VD=VD,VG=VG)
    assert not(np.any(np.isnan(If)) or np.any(np.isnan(Ib))), "NaNs!"

    plt.figure()
    plt.plot(VG,If)
    plt.plot(VG,Ib)
    plt.show()

    plt.yscale('log')

def test_weird_VO2():

    I_IMT=2.2e-3
    V_IMT=1.7
    I_MIT=1.2e-3
    V_MIT=.5
    R_met=0.01
    vo2=VO2Res(I_IMT=I_IMT,V_IMT=V_IMT,I_MIT=I_MIT,V_MIT=V_MIT,R_met=R_met)

    trans=vo2.I_IMT

    I=np.linspace(0,4*trans)
    plt.figure()
    plt.plot(vo2.V(I,Direction.FORWARD),I)
    plt.plot(vo2.V(I,Direction.BACKWARD),I)
    plt.show()

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
    Rs=Rd=0
    hemt=GaNHEMT_iMVGS(
        W=200e-6,Cinv_vxo=Cinv_vxo,
        VT0=VT0,alpha=alpha,SS=80e-3,delta=0,
        VDsats=VDsats,beta=beta,eta=0,Gleak=10**log10Gleak,Rs=Rs,Rd=Rd)
    vo2=VO2Res(I_IMT=I_IMT,V_IMT=V_IMT,I_MIT=I_MIT,V_MIT=V_MIT,R_met=R_met)

    hf=HyperFET(hemt,vo2)

    VD=np.array(50.0)
    VG=np.linspace(-7.5,0,500)
    #VG=np.linspace(-.5,0,10)

    #I=hemt.ID(VD=VD,VG=VG)
    #plt.plot(VG,I/hemt.W)


    If,Ib=[np.ravel(i) for i in hf.I_double(VD=VD,VG=VG)]
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
    hemt=GaNHEMT_iMVGS()
    vo2=VO2Res(R_met=1e-3)
    hf=HyperFET(hemt,vo2)

    numvalues=500
    Nrepeat=10
    VD=np.array(3.0)
    VG=np.linspace(-5,2,numvalues)

    #cProfile.runctx("hf.I(VD=VD,VG=VG,direc=Direction.FORWARD)",globals(),locals(),'restats.txt')
    hf_timing=timeit.timeit("hf.I_double(VD=VD,VG=VG)",number=Nrepeat,globals=locals())/Nrepeat
    print("\nHyperFET computed {:d} values in {:.3e} s (average of {:d} runs)." \
          .format(numvalues,hf_timing,Nrepeat))



if __name__=="__main__":
    test_weird_VO2()
    test_weird_HyperFET()
    pass
