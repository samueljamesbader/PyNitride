import matplotlib.pyplot as plt
import numpy as np
from pynitride import NodFunction

def is_Y(v):
    """ Is this mode dominantly along Y?

    Args:
        v: the mode as a NodFunction.
    Returns:
        bool, True if the largest component is Y
    """
    m=v.mesh
    dominant_component=np.argmax((m.density.tpf()*np.abs(v)**2).integrate(definite=True))
    return dominant_component==1

def is_AS(v):
    """ Is this mode AS (as opposed to SA)?

    For a mirror-symmetric heterostructure, the XZ modes should be either AS (X is antisymmetric, Z is symmetric)
    or SA (X is symmetric, Y is antisymmetric).  If modes are degenerate it can get a little tricky, but this function
    will return whether in some sense the mode seems AS.  If the mode is Y, then it will return False.

    Args:
        v: the mode as a NodFunction.
    Returns:
        bool, whether the mode seems mostly AS
    """
    m=v.mesh

    # Decide whether to look at the X or Z component by whichever is larger
    dominant_component=np.argmax((m.density.tpf()*np.abs(v)**2).integrate(definite=True))

    # Find the index of the biggest value of that component
    maxpoint=np.argmax(np.abs(v[dominant_component]))

    # Find the index of the point which is located most nearly opposite
    oppositepoint=m.indexn(m.zn[-1] - m.zn[maxpoint])

    # Take the ratio between the two values
    ratio=np.real(v[dominant_component,maxpoint]/v[dominant_component,oppositepoint])

    # The sign of the ratio determines whether this component is symmetric or assymetric
    is_domcomp_symm=ratio>0

    # Knowing which component this is and whether it's symmetric or not tells you if it's AS
    x=0; z=v.shape[0]-1;
    return ((dominant_component==x) and not is_domcomp_symm) or ((dominant_component==z) and is_domcomp_symm)


def sort_modes(en,vec,criteria):
    m=vec.mesh
    criteria+=[lambda v: True]
    bins=[[[],[]] for c in criteria]
    for iq in range(en.shape[0]):
        for be,bv in bins:
            be+=[[]];bv+=[[]]
        for eig in range(en.shape[1]):
            for ic,c in enumerate(criteria):
                if c(vec[iq,eig]):
                    bins[ic][0][iq]+=[en[iq,eig]]
                    bins[ic][1][iq]+=[vec[iq,eig]]
                    break
    for c in range(len(criteria)):
        countc=min(len(bins[c][0][iq]) for iq in range(len(en)))
        print("Criteria #",c," has ",countc," complete modes.")
        bins[c][0]=np.array([np.array(bins[c][0][iq][:countc]) for iq in range(len(en))])
        bins[c][1]=NodFunction(m,np.array([np.array(bins[c][1][iq][:countc]) for iq in range(len(en))],dtype='complex'),dtype='complex')
    return bins


def plot_mode(v):
    m=v.mesh
    plt.figure(figsize=(9,3))
    plt.subplot(131)
    plt.plot(m._zp,v[0].real,'.-')
    plt.plot(m._zp,v[0].imag)
    plt.axhline(0,color='k')
    plt.axvline(3,color='k')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.grid(True,axis='y')
    plt.subplot(132)
    plt.plot(m._zp,v[1].real)
    plt.plot(m._zp,v[1].imag)
    plt.axhline(0,color='k')
    plt.axvline(3,color='k')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.grid(True,axis='y')
    plt.subplot(133)
    plt.plot(m._zp,v[2].real)
    plt.plot(m._zp,v[2].imag)
    plt.axhline(0,color='k')
    plt.axvline(3,color='k')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.grid(True,axis='y')
    plt.tight_layout()
    plt.show()
    print(v)
