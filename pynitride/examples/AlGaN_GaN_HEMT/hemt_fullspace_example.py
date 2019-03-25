# Basic AlGaN/GaN HEMT
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np

from pynitride.carriers import MultibandKP
from pynitride.examples.AlGaN_GaN_HEMT.hemt_visualization import conduction_band_panels
from pynitride.examples.AlGaN_GaN_HEMT.hemt_example import define_mesh as define_basic_mesh
from pynitride.maths import dephase
from pynitride.paramdb import to_unit, nm
from pynitride.reciprocal_mesh import RMesh1D, RMesh2D_Polar
from pynitride.sim import Simulation


@wraps(define_basic_mesh)
def define_mesh(sim,*args,**kwargs):
    define_basic_mesh(sim,*args,**kwargs)
    sim.rmeshes['mbkp']=RMesh1D.regular(2/nm,20)
    sim.dmeshes['mbkp']=sim.dmeshes['schro']
    del sim.dmeshes['schro']

if __name__=="__main__":
    sim=Simulation("HEMT",define_mesh,Simulation.flow_semiclassicalramp_mbkp,
        solve_opts={'mbkp_opts':{'carriers':['electron']}})
    sim.load(force=True)

    m,schro,mbkp_2DEG=sim.dmeshes['main'],sim.dmeshes['mbkp'],sim.extras['mbkp']
    m.plot_mesh()
    plt.tight_layout()


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

