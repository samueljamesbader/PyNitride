# GaN/AlN p-channel HFET

#from pynitride.machine import Pool;Pool.configure_onlycextparallel();

import matplotlib.pyplot as plt
from pynitride.mesh import Mesh, PointFunction, MidFunction, MaterialBlock, UniformLayer
from pynitride.material import AlGaN
from pynitride.paramdb import to_unit, nm, eV, m_e, cm
from pynitride.carriers import Schrodinger, Semiclassical, MultibandKP
from pynitride.solvers import PoissonSolver, Equilibrium, SelfConsistentLoop
from pynitride.thermal import ConstantT
from pynitride.strain import Pseudomorphic
from examples.pchannel_GaN_AlN_HFET.pFET_visualization import valence_band_panels
from time import time
import numpy as np


if __name__=="__main__":


    # Set up the mesh
    well_t=15*nm
    buff_t=200*nm
    m=Mesh([
        MaterialBlock("epi",AlGaN(),[
            UniformLayer("well"  ,  well_t, x=0, DeepDonorDonorConc=5e16/cm**3),
            UniformLayer("buffer",  buff_t, x=1, DeepDonorDonorConc=5e16/cm**3),
        ])],
        max_dz=5*nm,
        refinements=[[0,.05*nm,3],['well/buffer',.01*nm,1.4]],uniform=False,boundary=[.7*eV,"thick"])

    print("Mesh points: ",m.Np)
    schro,semi=m.submesh_cover([well_t+5*nm])


    Equilibrium(m)
    ConstantT(m)
    Pseudomorphic(m)
    ps=PoissonSolver(m)

    scl=SelfConsistentLoop(
        fieldsolvers=[PoissonSolver(m)],
        carriermodels=[Semiclassical  (schro,carriers=['hole']),
                        Semiclassical(schro,carriers=['electron']),
                        Semiclassical(semi)])
    scl.ramp_epsfactor(start=1e4, max_iter=20, dlefmin=.005, tol=1e-5)

    starttime=time()
    from pynitride.reciprocal_mesh import RMesh1D, RMesh2D_Polar
    #rmesh=RMesh1D.regular(kmax=2/nm,numabsk=25)
    rmesh=RMesh2D_Polar.regular(kmax=2.5/nm,numabsk=24,numtheta=4,align_theta=True,d=4)
    mbkp=scl._cs[0]=MultibandKP(schro,num_eigenvalues=6,rmesh=rmesh)
    scl.loop(tol=1e-5,min_activation=.05)
    #scl.loop(tol=1e5,min_activation=.05)
    endtime=time()
    print("kp loop took {:.1f} sec".format(endtime-starttime))

    print("Holes: {:.2f} x10^13/cm^2".format(to_unit(float(m.p.integrate(definite=True)),"1e13/cm^2")))
    print("EV-EF [meV]: {:.2f} meV",to_unit(float((m.Ev-m.EF.tmf())[m.indexm(well_t)]),"meV"))

    # Check normalization
    wf0=rmesh['kppsi'][0,0,:,:]
    wf1=rmesh['kppsi'][0,1,:,:]
    wf2=rmesh['kppsi'][0,2,:,:]
    from pynitride.mesh import inner_product
    assert np.isclose(inner_product(wf0,wf0),1,atol=1e-8)
    assert np.isclose(inner_product(wf0,wf1),0,atol=1e-8)
    assert np.isclose(inner_product(wf0,wf2),0,atol=1e-8)
    assert np.isclose(inner_product(wf1,wf2),0,atol=1e-8)
    assert np.isclose(inner_product(wf1,wf1),1,atol=1e-8)
    assert np.isclose(inner_product(wf2,wf2),1,atol=1e-8)

    if 0:
        plt.figure()
        plt.plot(m.zm,m.Ec,'b')
        plt.plot(m.zm,m.Ev,'g')
        plt.plot(m.zp,m.EF,'r')
        i=1
        plt.plot(schro.zp,rmesh['normsqs'][0,0,:]+rmesh['kpen'][0,0],'purple')
        plt.plot(schro.zp,rmesh['normsqs'][0,2,:]+rmesh['kpen'][0,2],'pink')
        plt.plot(schro.zp,rmesh['normsqs'][0,4,:]+rmesh['kpen'][0,2],'black')
        plt.twinx()
        plt.fill_between(m.zp,m.p,color='b',alpha=.2)
        plt.xlim(0,50*nm)
        plt.ylim(0)
        plt.figure()
        plt.plot(rmesh.kx[rmesh.ky==0],rmesh['kpen'][rmesh.ky==0][:,0]*1e3)
        plt.plot(rmesh.kx[rmesh.ky==0],rmesh['kpen'][rmesh.ky==0][:,2]*1e3)
        plt.plot(rmesh.kx[rmesh.ky==0],rmesh['kpen'][rmesh.ky==0][:,4]*1e3)
        plt.axhline(0,color='k')
        plt.xlim(0)
        plt.show()

    valence_band_panels(m,mbkp)
    plt.show()
