# GaN/AlN p-channel HFET

#from pynitride.machine import Pool;Pool.configure_onlycextparallel();

import matplotlib.pyplot as plt
import numpy as np

from pynitride.examples.pchannel_GaN_AlN_HFET.pFET_visualization import valence_band_panels
from pynitride.physics.material import AlGaN
from pynitride import Mesh, MaterialBlock, UniformLayer
from pynitride import to_unit, nm, eV, cm, meV
from pynitride import RMesh2D_Polar, RMesh1D
from pynitride import Simulation
from pynitride import log

from pynitride.examples.pchannel_GaN_AlN_HFET.pFET_example import define_mesh

if __name__=="__main__":

    sim=Simulation('pFET',define_mesh=define_mesh,
       #mesh_opts={'kmesh':'1D'},
       solve_flow=Simulation.flow_semiclassicalramp_mbkp,
       solve_opts ={'T':300.0,'ramp_T':10,'mbkp_opts':{'num_eigenvalues':6},'Va':4,
                    'mbkp_loop_opts':{'init_activation':.1, 'inc_activation':1.3},
                    'Tramp_loop_opts':{'init_activation':.5,'inc_activation':1.3,'min_activation':.005}})
    sim.load(force=True)
    m,quantum=sim.dmeshes['main'],sim.dmeshes['mbkp']
    rmesh=sim.rmeshes['mbkp_out']


    print("Holes: {:.2f} x10^13/cm^2".format(to_unit(float(m.p.integrate(definite=True)),"1e13/cm^2")))
    print("EV-EF [meV]: {:.2f} meV".format(to_unit(float((m.Ev-m.EF.tmf())[m.indexm(sim.extras['well_t'])]),"meV")))

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
        plt.plot(m.zn, m.EF, 'r')
        i=1
        plt.plot(quantum.zn, rmesh['normsqs'][0, 0, :] + rmesh['kpen'][0, 0], 'purple')
        plt.plot(quantum.zn, rmesh['normsqs'][0, 2, :] + rmesh['kpen'][0, 2], 'pink')
        plt.plot(quantum.zn, rmesh['normsqs'][0, 4, :] + rmesh['kpen'][0, 2], 'black')
        plt.twinx()
        plt.fill_between(m.zn, m.p, color='b', alpha=.2)
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
