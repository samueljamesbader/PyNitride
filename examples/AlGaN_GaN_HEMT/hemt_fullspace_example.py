# Basic AlGaN/GaN HEMT
import matplotlib.pyplot as plt
from pynitride.mesh import Mesh, PointFunction, MidFunction, MaterialBlock, UniformLayer
from pynitride.material import AlGaN
from pynitride.paramdb import to_unit, nm, eV, m_e, cm
from pynitride.carriers import MultibandKP, Semiclassical
from pynitride.reciprocal_mesh import RMesh1D, RMesh2D_Polar
from pynitride.solvers import PoissonSolver, Equilibrium, SelfConsistentLoop
from pynitride.thermal import ConstantT
from pynitride.strain import Pseudomorphic
from pynitride.maths import dephase
from examples.AlGaN_GaN_HEMT.hemt_visualization import conduction_band_panels
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

    semi_2DEG=Semiclassical(schro,carriers=['electron'])
    scl=SelfConsistentLoop(
        fieldsolvers=[PoissonSolver(m)],
        carriermodels=[semi_2DEG,
                        Semiclassical(schro,carriers=['hole']),
                        Semiclassical(semi)])
    scl.ramp_epsfactor(start=1e3, max_iter=20, dlefmin=.005, tol=1e-5)
    rmesh=RMesh1D.regular(2/nm,20)
    mbkp_2DEG=MultibandKP(schro,rmesh=rmesh,carriers=['electron'],num_eigenvalues=20)
    scl.swap_carrier_model(remove=semi_2DEG,add=mbkp_2DEG)
    scl.loop(tol=1e-5)

    # Check normalization
    wf0=mbkp_2DEG.kppsi[0,0]
    wf1=mbkp_2DEG.kppsi[0,1]
    wf2=mbkp_2DEG.kppsi[0,2]
    from pynitride.mesh import inner_product
    assert np.isclose(inner_product(wf0,wf0),1,atol=1e-8)
    assert np.isclose(inner_product(wf0,wf1),0,atol=1e-8)
    assert np.isclose(inner_product(wf0,wf2),0,atol=1e-8)
    print("Total electrons: {:.2g} x10^13/cm^2".format(to_unit(float(m.n.integrate(definite=True)),"1e13/cm^2")))

    if 1:
        plt.figure()
        plt.plot(m.zm,m.Ec,'b')
        plt.plot(m.zm,m.Ev,'g')
        plt.plot(m.zp,m.EF,'r')
        plt.plot(schro.zp,dephase(np.sum(mbkp_2DEG.kppsi[0,0,:,:],axis=0))+mbkp_2DEG.kpen[0,0],'purple')
        plt.plot(schro.zp,dephase(np.sum(mbkp_2DEG.kppsi[0,2,:,:],axis=0))+mbkp_2DEG.kpen[0,2],'pink')
        plt.plot(schro.zp,dephase(np.sum(mbkp_2DEG.kppsi[0,4,:,:],axis=0))+mbkp_2DEG.kpen[0,4],'black')
        plt.ylabel("Energy [eV]")
        plt.xlabel("Depth [nm]")
        plt.twinx()
        plt.fill_between(m.zp,m.n,color='b',alpha=.2)
        plt.xlim(0,50*nm)
        plt.ylim(0)
        plt.yticks([])
        plt.tight_layout()
        #plt.show()


    rmesh=RMesh2D_Polar.regular(2/nm,20,4,align_theta=True)
    mbkp_2DEG=MultibandKP(schro,rmesh=rmesh,carriers=['electron'],num_eigenvalues=20)
    mbkp_2DEG.solve()
    conduction_band_panels(m,mbkp_2DEG)
    plt.show()

