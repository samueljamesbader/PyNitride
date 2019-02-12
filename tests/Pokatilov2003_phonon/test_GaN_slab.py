""" Solves the GaN slab of
`Pokatilov 2003 <https://doi.org/10.1016/S0749-6036(03)00069-7>`_
for comparison."""


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pynitride.mesh import Mesh, MaterialBlock, UniformLayer
from pynitride.paramdb import to_unit
from pynitride.material import AlGaN
from pynitride.paramdb import nm, eV, m_e
from pynitride.phonons import ElasticContinuum
from pynitride.reciprocal_mesh import RMesh1D
import numpy as np
from tests.Pokatilov2003_phonon.phonon_analysis import sort_modes, is_Y, is_AS

if __name__=="__main__":
    m=Mesh([
        MaterialBlock("slab",AlGaN(),[
            UniformLayer("slab", 6*nm, x=0),
        ])],
        max_dz=.025*nm,
        refinements=[],uniform=True)
    print("Mesh points: ",m.Np)

    ec=ElasticContinuum(m,num_eigs=40,rmesh=RMesh1D.regular(2*np.pi,100,.005/nm))
    ec.solve()

    (y_en,y_vec),(as_en,as_vec),(sa_en,sa_vec)=sort_modes(ec._en,ec._vecs, [is_Y, is_AS])

    plt.figure()
    plt.plot(ec.q,to_unit(y_en[:,:6],"meV"),'k')
    plt.xlim(0,6.31)
    plt.ylim(0,20.3)
    plt.gca().yaxis.set_major_locator(MultipleLocator(4))
    plt.xticks(np.linspace(.31,6.26,num=6))
    plt.grid(True)
    plt.title("Y-modes (Pokatilov 1a)")
    plt.xlabel("Wavevector [nm$^{-1}$]")
    plt.ylabel("Energy [meV]")

    plt.figure()
    plt.plot(ec.q,to_unit(sa_en[:,:6],"meV"),'k')
    plt.plot(ec.q,to_unit(as_en[:,:6],"meV"),'--k')
    plt.xlim(0,6.31)
    plt.ylim(0,27.8)
    plt.gca().yaxis.set_major_locator(MultipleLocator(5))
    plt.xticks(np.linspace(.31,6.26,num=6))
    plt.grid(True)
    plt.title("XZ-modes (Pokatilov 4d)")
    plt.xlabel("Wavevector [nm$^{-1}$]")
    plt.ylabel("Energy [meV]")
    plt.show()

