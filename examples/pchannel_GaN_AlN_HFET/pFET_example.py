# GaN/AlN p-channel HFET

from pynitride.machine import Pool; Pool.configure_onlycextparallel();

import matplotlib.pyplot as plt
from pynitride.mesh import Mesh, PointFunction, MidFunction, MaterialBlock, UniformLayer
from pynitride.material import AlGaN
from pynitride.paramdb import to_unit, nm, eV, m_e, cm
from pynitride.carriers import Schrodinger, Semiclassical, MultibandKP
from pynitride.solvers import PoissonSolver, Equilibrium, SelfConsistentLoop
from pynitride.thermal import ConstantT
from pynitride.strain import Pseudomorphic
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
        refinements=[['well/buffer',.01*nm,1.4]],uniform=False,boundary=[.7*eV,"thick"])

    schro,semi=m.submesh_cover([well_t+5*nm])

    print("Mesh points: ",m.Np)
    #m.plot_mesh()
    #plt.show()

    Equilibrium(m)
    ConstantT(m)
    Pseudomorphic(m)
    ps=PoissonSolver(m)

    scl=SelfConsistentLoop(
        fieldsolvers=[PoissonSolver(m)],
        carriersolvers=[Semiclassical  (schro,carriers=['hole']),
                        Semiclassical(schro,carriers=['electron']),
                        Semiclassical(semi)])
    scl.ramp_epsfactor(start=1e4, max_iter=20, dlefmin=.005, tol=1e-5)

    starttime=time()
    mbkp=scl._cs[0]=MultibandKP(schro,num_eigenvalues=6)
    scl.loop(tol=1e-5,min_activation=.05)
    endtime=time()
    print("kp loop took {:.1f} sec".format(endtime-starttime))

    print("Holes: {:.2f} x10^13/cm^2".format(to_unit(float(m.p.integrate(definite=True)),"1e13/cm^2")))

    # Check normalization
    wf0=mbkp._kppsi[0,0,:,:]
    wf1=mbkp._kppsi[0,1,:,:]
    wf2=mbkp._kppsi[0,2,:,:]
    from pynitride.mesh import inner_product
    assert np.isclose(inner_product(wf0,wf0),1,atol=1e-8)
    assert np.isclose(inner_product(wf0,wf1),0,atol=1e-8)
    assert np.isclose(inner_product(wf0,wf2),0,atol=1e-8)
    assert np.isclose(inner_product(wf1,wf2),0,atol=1e-8)
    assert np.isclose(inner_product(wf1,wf1),1,atol=1e-8)
    assert np.isclose(inner_product(wf2,wf2),1,atol=1e-8)


    plt.figure()
    plt.plot(m.zm,m.Ec,'b')
    plt.plot(m.zm,m.Ev,'g')
    plt.plot(m.zp,m.EF,'r')
    i=1
    if hasattr(mbkp,'_hpsi'):
        plt.plot(schro.zp,mbkp._hpsi[i,0,:]+mbkp._hen[i,0],'purple')
        plt.plot(schro.zp,mbkp._hpsi[i,1,:]+mbkp._hen[i,1],'pink')
        plt.plot(schro.zp,mbkp._hpsi[i,2,:]+mbkp._hen[i,2],'black')
    if hasattr(mbkp,'_normsqs'):
        plt.plot(schro.zp,mbkp._normsqs[0,0,:]+mbkp._kpen[0,0],'purple')
        plt.plot(schro.zp,mbkp._normsqs[0,2,:]+mbkp._kpen[0,2],'pink')
        plt.plot(schro.zp,mbkp._normsqs[0,4,:]+mbkp._kpen[0,2],'black')
        #plt.plot(schro.zp,mbkp._normsqs[0,6,:]+mbkp._kpen[0,2],'black')
    plt.twinx()
    plt.fill_between(m.zp,m.p,'b',alpha=.2)
    plt.xlim(0,50*nm)
    plt.ylim(0)
    plt.figure()
    plt.plot(mbkp._kx,mbkp._kpen[:,0]*1e3)
    plt.plot(mbkp._kx,mbkp._kpen[:,2]*1e3)
    plt.plot(mbkp._kx,mbkp._kpen[:,4]*1e3)
    #plt.plot(mbkp._kx,mbkp._kpen[:,6]*1e3)
    plt.axhline(0,color='k')
    plt.xlim(0)
    plt.show()

