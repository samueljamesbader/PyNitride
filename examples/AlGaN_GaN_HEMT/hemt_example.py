# Basic AlGaN/GaN HEMT
import matplotlib.pyplot as plt
from pynitride.mesh import Mesh, PointFunction, MidFunction, MaterialBlock, UniformLayer
from pynitride.material import AlGaN
from pynitride.paramdb import nm, eV, m_e, cm
from pynitride.carriers import Schrodinger, Semiclassical
from pynitride.solvers import PoissonSolver, Equilibrium, SelfConsistentLoop
from pynitride.thermal import ConstantT
from pynitride.strain import Pseudomorphic
import numpy as np


if __name__=="__main__":


    # Set up the mesh
    barr_t=20*nm
    barr_x=.4
    buff_t=100*nm
    m=Mesh([
        MaterialBlock("epi",AlGaN(),[
            UniformLayer("barrier",  barr_t, x=barr_x, DeepDonorDonorConc=5e16/cm**3),
            UniformLayer("buffer" ,  buff_t, x=     0, DeepDonorDonorConc=5e16/cm**3),
        ])],
        max_dz=1*nm,
        refinements=[['barrier/buffer',.01*nm,1.4]],uniform=False,boundary=[.7*eV,"thick"])

    schro,semi=m.submesh_cover([barr_t+30*nm])

    print("Mesh points: ",m.Np)
    m.plot_mesh()
    plt.tight_layout()

    Equilibrium(m)
    ConstantT(m)
    Pseudomorphic(m)
    ps=PoissonSolver(m)

    scl=SelfConsistentLoop(
        fieldsolvers=[PoissonSolver(m)],
        carriermodels=[Schrodinger  (schro,carriers=['electron'],num_eigenvalues=20),
                        Semiclassical(schro,carriers=['hole']),
                        Semiclassical(semi)])
    scl.ramp_epsfactor(start=1e3, max_iter=20, dlefmin=.005, tol=1e-5)

    # Check normalization
    wf0=scl._cs[0]._epsi[0,0,:]
    wf1=scl._cs[0]._epsi[0,1,:]
    from pynitride.mesh import inner_product
    assert np.isclose(inner_product(wf0,wf0),1,atol=1e-8)
    assert np.isclose(inner_product(wf0,wf1),0,atol=1e-8)

    if 1:
        plt.figure()
        plt.plot(m.zm,m.Ec,'b')
        plt.plot(m.zm,m.Ev,'g')
        plt.plot(m.zp,m.EF,'r')
        plt.plot(schro.zp,scl._cs[0]._epsi[0,0,:]+scl._cs[0]._een[0,0],'purple')
        plt.plot(schro.zp,scl._cs[0]._epsi[0,1,:]+scl._cs[0]._een[0,1],'pink')
        plt.plot(schro.zp,scl._cs[0]._epsi[0,2,:]+scl._cs[0]._een[0,2],'black')
        plt.ylabel("Energy [eV]")
        plt.xlabel("Depth [nm]")
        plt.twinx()
        plt.fill_between(m.zp,m.n,color='b',alpha=.2)
        plt.xlim(0,50*nm)
        plt.ylim(0)
        plt.yticks([])
        plt.tight_layout()
        plt.show()

