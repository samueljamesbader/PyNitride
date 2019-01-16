""" Solves the thin sandwich of `Pokatilov 2003 <https://doi.org/10.1016/S0749-6036(03)00069-7>`_ for comparison
using the just XZ modes option."""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pynitride.mesh import Mesh, MaterialBlock, UniformLayer
from pynitride.paramdb import to_unit
from pynitride.material import AlGaN
from pynitride.paramdb import nm, eV, m_e
from pynitride.phonons import ElasticContinuum
import numpy as np
from tests.Pokatilov2003_phonon.phonon_analysis import sort_modes, is_Y, is_AS

if __name__=="__main__":
    m=Mesh([
        MaterialBlock("slab",AlGaN(),[
            UniformLayer("top", 2.5*nm, x=1),
            UniformLayer("middle", 1*nm, x=0),
            UniformLayer("bottom", 2.5*nm, x=1),
        ])],
        max_dz=.025*nm,
        refinements=[],uniform=True)
    print("Mesh points: ",m.Np)

    ec=ElasticContinuum(m,num_eigenvalues=40,qmax=2*np.pi,num_qpoints=100,qshift=.005/nm,justXZ=True)
    ec.solve()

    (as_en,as_vec),(sa_en,sa_vec)=sort_modes(ec._en,ec._vecs, [is_AS])

    plt.figure()
    plt.plot(ec._q,to_unit(sa_en[:,:6],"meV"),'k')
    plt.plot(ec._q,to_unit(as_en[:,:6],"meV"),'--k')
    plt.xlim(0,6.31)
    plt.ylim(0,34.83)
    plt.gca().yaxis.set_major_locator(MultipleLocator(4))
    plt.xticks(np.linspace(.31,6.26,num=6))
    plt.grid(True)
    plt.title("XZ-modes (Pokatilov 4b)")
    plt.xlabel("Wavevector [nm$^{-1}$]")
    plt.title("XZ-modes (Pokatilov 4b)")
    plt.show()

