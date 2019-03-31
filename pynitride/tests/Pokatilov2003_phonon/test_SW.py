""" Solves the thin sandwich of `Pokatilov 2003 <https://doi.org/10.1016/S0749-6036(03)00069-7>`_ for comparison."""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pynitride import Mesh, MaterialBlock, UniformLayer
from pynitride import to_unit, nm
from pynitride.physics.material import AlGaN
from pynitride.physics.phonons import ElasticContinuum
from pynitride import RMesh1D
import numpy as np
from pynitride.tests.Pokatilov2003_phonon.phonon_analysis import sort_modes, is_Y, is_AS
from time import time

if __name__=="__main__":
    starttime=time()
    m=Mesh([
        MaterialBlock("slab",AlGaN(),[
            UniformLayer("top", 2.5*nm, x=1),
            UniformLayer("middle", 1*nm, x=0),
            UniformLayer("bottom", 2.5*nm, x=1),
        ])],
        max_dz=.025*nm,
        refinements=[],uniform=True)
    print("Mesh points: ",m.Np)

    ec=ElasticContinuum(m,num_eigenvalues=40,rmesh=RMesh1D.regular(2*np.pi,100,.005/nm))
    ec.solve()
    endtime=time()
    print("Took {:.2g} sec".format(endtime-starttime))

    (y_en,y_vec),(as_en,as_vec),(sa_en,sa_vec)=sort_modes(ec.en,ec.vecs, [is_Y, is_AS])

    plt.figure()
    plt.plot(ec.q,to_unit(y_en[:,:6],"meV"),'k')
    plt.xlim(0,6.31)
    plt.ylim(-1,27.93)
    plt.gca().yaxis.set_major_locator(MultipleLocator(4))
    plt.xticks(np.linspace(.31,6.26,num=6))
    plt.grid(True)
    plt.title("Y-modes (Pokatilov 1b)")
    plt.xlabel("Wavevector [nm$^{-1}$]")
    plt.ylabel("Energy [meV]")

    plt.figure()
    plt.plot(ec.q,to_unit(sa_en[:,:6],"meV"),'k')
    plt.plot(ec.q,to_unit(as_en[:,:6],"meV"),'--k')
    plt.xlim(0,6.31)
    plt.ylim(-1,34.83)
    plt.gca().yaxis.set_major_locator(MultipleLocator(4))
    plt.xticks(np.linspace(.31,6.26,num=6))
    plt.grid(True)
    plt.title("XZ-modes (Pokatilov 4b)")
    plt.xlabel("Wavevector [nm$^{-1}$]")
    plt.ylabel("Energy [meV]")
    plt.show()

