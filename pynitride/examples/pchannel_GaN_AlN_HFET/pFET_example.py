# GaN/AlN p-channel HFET

#from pynitride.machine import Pool;Pool.configure_onlycextparallel();

import matplotlib.pyplot as plt
import numpy as np

from pynitride.examples.pchannel_GaN_AlN_HFET.pFET_visualization import valence_band_panels
from pynitride.physics.material import AlGaN
from pynitride import Mesh, MaterialBlock, UniformLayer
from pynitride import to_unit, nm, eV, cm, meV
from pynitride import RMesh2D_Polar
from pynitride import Simulation
from pynitride import log


def define_mesh(sim,well_t=15*nm,buff_t=200*nm,Ndd=5e16/cm**3,max_dz=5*nm,sbh=1.4*eV,ss=0*meV):

    # Set up the main mesh
    m=sim.dmeshes['main']=Mesh([
        MaterialBlock("epi",AlGaN(spin_splitting=ss),[
            UniformLayer("well"  ,  well_t, x=0, DeepDonorDonorConc=Ndd),
            UniformLayer("buffer",  buff_t, x=1, DeepDonorDonorConc=Ndd),
        ])],
        max_dz=max_dz,
        refinements=[[0,.03*nm,2],['well/buffer',.01*nm,1.5]],
        uniform=False,boundary=[sbh,"thick"])
    log("Mesh points "+str(m.Np))

    # Set up a quantum mesh
    sim.dmeshes['mbkp'],sim.dmeshes['semi']=m.submesh_cover([well_t+5*nm],['mbkp','semi'])

    # Set up the reciprocal space mesh for MBKP
    sim.rmeshes['mbkp_solve']=RMesh2D_Polar.regular(kmax=2.5/nm,numabsk=24,numtheta=4,align_theta=True,d=1)
    sim.rmeshes['mbkp_out'  ]=RMesh2D_Polar.regular(kmax=4.8/nm,numabsk=48,numtheta=4,align_theta=True,d=1)

    sim.extras['well_t']=well_t
    sim.extras['sourcepoint']=float(well_t-2.5)

if __name__=="__main__":

    sim=Simulation('pFET',define_mesh=define_mesh,
       solve_flow=Simulation.flow_semiclassicalramp_mbkp,
       solve_opts ={'mbkp_opts':{'num_eigenvalues':6},'Va':4})
    sim.load(force=True)
    m,quantum=sim.dmeshes['main'],sim.dmeshes['mbkp']
    rmesh=sim.rmeshes['mbkp_out']


    print("Holes: {:.2f} x10^13/cm^2".format(to_unit(float(m.p.integrate(definite=True)),"1e13/cm^2")))
    print("EV-EF [meV]: {:.2f} meV",to_unit(float((m.Ev-m.EF.tmf())[m.indexm(sim.extras['well_t'])]),"meV"))

    # Check normalization
    if 'kppsi' in rmesh:
        wf0=rmesh['kppsi'][0,0,:,:]
        wf1=rmesh['kppsi'][0,1,:,:]
        wf2=rmesh['kppsi'][0,2,:,:]
        from pynitride.core.mesh import inner_product
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
        plt.plot(quantum.zp,rmesh['normsqs'][0,0,:]+rmesh['kpen'][0,0],'purple')
        plt.plot(quantum.zp,rmesh['normsqs'][0,2,:]+rmesh['kpen'][0,2],'pink')
        plt.plot(quantum.zp,rmesh['normsqs'][0,4,:]+rmesh['kpen'][0,2],'black')
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

    from pynitride.physics.carriers import MultibandKP
    mbkp=MultibandKP(quantum,rmesh,num_eigenvalues=6)
    valence_band_panels(m,mbkp)
    plt.show()
