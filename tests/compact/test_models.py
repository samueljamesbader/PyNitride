from pynitride.compact.models import GaNHEMT_iMVGS, VO2Res, Direction, HyperFET
import timeit
import numpy as np
import matplotlib.pyplot as plt
import pytest
import cProfile

if __name__=="__main__":
    pytest.main(args=[__file__,"-s","--plots"])

# def test_ganhemt():
#     hemt=GaNHEMT_iMVGS()
#
#     VD=np.array(3.0)
#     VG=np.linspace(-7,2)
#     I=hemt.ID(VD=VD,VG=VG)
#     #Ff=[hemt.Ff(VDS=VD,VGS=vg) for vg in VG]
#     plt.figure()
#     plt.plot(VG,I)
#     #plt.plot(VG,Ff)
#     plt.show()
#     plt.gcf().canvas.set_window_title("HEMT")
#     plt.yscale('log')
#
#     #print(I)
#     #print(Ff)


# def test_VO2_Res():
#     vo2=VO2Res(R_met=1e-3)
#
#     trans=vo2.I_IMT
#
#     I=np.linspace(0,4*trans)
#     plt.figure()
#     plt.plot(vo2.V(I,Direction.FORWARD),I)
#     plt.plot(vo2.V(I,Direction.BACKWARD),I)
#     plt.show()
#
#     #I=np.linspace(fi_min,fi_max*4)
#     #V=vo2.V(I)
#     #plt.figure()
#     #plt.plot(V,I)
#     #plt.show()

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
