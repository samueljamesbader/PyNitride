# Basic AlGaN/GaN HEMT
import matplotlib.pyplot as plt
from pynitride.mesh import Mesh, PointFunction, MidFunction, MaterialBlock, UniformLayer
from pynitride.material import AlGaN
from pynitride.paramdb import to_unit, nm, eV, m_e, cm
from pynitride.carriers import Schrodinger, Semiclassical
from pynitride.solvers import PoissonSolver, Equilibrium, SelfConsistentLoop
from pynitride.thermal import ConstantT
from pynitride.strain import Pseudomorphic
from pynitride.maths import dephase
from pynitride.visual import log
from pynitride.sim import Simulation
import numpy as np

def define_mesh(sim,barr_t=20*nm,barr_x=.4,buff_t=100*nm,Ndd=5e16/cm**3,max_dz=1*nm):

    # Set up the mesh
    sim.dmeshes['main']=m=Mesh([
        MaterialBlock("epi",AlGaN(),[
            UniformLayer("barrier",  barr_t, x=barr_x, DeepDonorDonorConc=5e16/cm**3),
            UniformLayer("buffer" ,  buff_t, x=     0, DeepDonorDonorConc=5e16/cm**3),
        ])],
        max_dz=max_dz,
        refinements=[['barrier/buffer',.01*nm,1.4]],uniform=False,boundary=[.7*eV,"thick"])
    log("Mesh points: "+str(m.Np))

    sim.dmeshes['schro'],sim.dmeshes['semi']=\
        m.submesh_cover([barr_t+30*nm],names=['schro','semi'])

if __name__=="__main__":

    sim=Simulation("HEMT",define_mesh,Simulation.flow_semiclassicalramp_schrodinger)
    sim.load(force=True)

    m,schro=sim.dmeshes['main'],sim.dmeshes['schro']
    ss=sim.extras['schro']
    m.plot_mesh()
    plt.tight_layout()


    # Check normalization
    wf0=ss._epsi[0,0,:]
    wf1=ss._epsi[0,1,:]
    from pynitride.mesh import inner_product
    assert np.isclose(inner_product(wf0,wf0),1,atol=1e-8)
    assert np.isclose(inner_product(wf0,wf1),0,atol=1e-8)
    print("Total electrons: {:.2g} x10^13/cm^2".format(to_unit(float(m.n.integrate(definite=True)),"1e13/cm^2")))

    if 1:
        plt.figure()
        plt.plot(m.zm,m.Ec,'b')
        plt.plot(m.zm,m.Ev,'g')
        plt.plot(m.zp,m.EF,'r')
        plt.plot(schro.zp,dephase(ss._epsi[0,0,:])+ss._een[0,0],'purple')
        plt.plot(schro.zp,dephase(ss._epsi[0,1,:])+ss._een[0,1],'pink')
        plt.plot(schro.zp,dephase(ss._epsi[0,2,:])+ss._een[0,2],'black')
        plt.ylabel("Energy [eV]")
        plt.xlabel("Depth [nm]")
        plt.twinx()
        plt.fill_between(m.zp,m.n,color='b',alpha=.2)
        plt.xlim(0,50*nm)
        plt.ylim(0)
        plt.yticks([])
        plt.tight_layout()
        plt.show()

