from examples.pchannel_GaN_AlN_HFET.pFET_example import define_mesh as define_electrical_mesh
from pynitride.phonons import DielectricContinuum_SWH
from pynitride.reciprocal_mesh import RMesh1D
from pynitride.paramdb import to_unit, nm, meV, hbar
from pynitride.sim import Simulation
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

def define_mesh(sim,**kwargs):
    define_electrical_mesh(sim,**kwargs)
    sim.rmeshes['POP']=RMesh1D.regular(1/nm,100,abskshift=.001/nm)

def solve_flow(sim):
    m=sim.dmeshes['main']
    rmesh=sim.rmeshes['POP']
    dc=DielectricContinuum_SWH(m,rmesh,
        {'TOu':5,'TOIF':1,'TOl':10,'LOu':5,'LOIF':1,'LOl':10})
    dc.solve(just_energies=False)
    sim.extras['dc']=dc

def POP_panel(dc):
    plt.figure(figsize=(4,8))
    ax_meV=plt.gca()

    wLO_perp_G,wLO_para_G,wLO_perp_A,wLO_para_A,\
    wTO_perp_G,wTO_para_G,wTO_perp_A,wTO_para_A,\
    epsinf_G,epsinf_A,tw,tb=dc._params


    # Plot the solution
    plt.plot(dc.rmesh.absk,to_unit(dc.en(),"meV"))

    plt.fill_between([0,dc.q[-1]],to_unit(hbar*wTO_para_G,"meV"),color='k',alpha=.25)
    plt.fill_between([0,dc.q[-1]],to_unit(hbar*wLO_perp_A,"meV"),1000,color='k',alpha=.25)
    plt.fill_between([0,dc.q[-1]],to_unit(hbar*wTO_perp_A,"meV"),to_unit(hbar*wLO_para_G,"meV"),color='k',alpha=.25)

    # Style the meV axes
    plt.ylim(65,125)
    plt.ylabel("Energy [meV]")


    lltrans=transforms.blended_transform_factory(plt.gca().transAxes,plt.gca().transData,)
    wlabelsize=10
    plt.axhline(to_unit( hbar*wTO_para_A,'meV'),color='k',linestyle='--',linewidth=2)
    plt.text(1,to_unit(hbar*wTO_para_A,'meV'),
             r' $\omega_{TO}\parallel\,$ AlN',transform=lltrans,fontsize=wlabelsize,ha='left',va='center')
    plt.axhline(to_unit( hbar*wTO_perp_A,'meV'),color='k',linestyle='--',linewidth=2)
    plt.text(1,to_unit(hbar*wTO_perp_A,'meV'),
             r' $\omega_{TO}\perp    $ AlN',transform=lltrans,fontsize=wlabelsize,ha='left',va='center')

    plt.axhline(to_unit( hbar*wTO_para_G,'meV'),color='k',linestyle='--',linewidth=2)
    plt.text(1,to_unit(hbar*wTO_para_G,'meV')
             ,r' $\omega_{TO}\parallel\,$ GaN',transform=lltrans,fontsize=wlabelsize,ha='left',va='center')
    plt.axhline(to_unit( hbar*wTO_perp_G,'meV'),color='k',linestyle='--',linewidth=2)
    plt.text(1,to_unit(hbar*wTO_perp_G,'meV')
             ,r' $\omega_{TO}\perp$ GaN',transform=lltrans,fontsize=wlabelsize,ha='left',va='center')

    plt.axhline(to_unit( hbar*wLO_para_A,'meV'),color='k',linestyle='--',linewidth=2)
    plt.text(1,to_unit(hbar*wLO_para_A,'meV')
             ,r' $\omega_{LO}\parallel\,$ AlN',transform=lltrans,fontsize=wlabelsize,ha='left',va='center')
    plt.axhline(to_unit( hbar*wLO_perp_A,'meV'),color='k',linestyle='--',linewidth=2)
    plt.text(1,to_unit(hbar*wLO_perp_A,'meV')
             ,r' $\omega_{LO}\perp$ AlN',transform=lltrans,fontsize=wlabelsize,ha='left',va='center')

    plt.axhline(to_unit( hbar*wLO_para_G,'meV'),color='k',linestyle='--',linewidth=2)
    plt.text(1,to_unit(hbar*wLO_para_G,'meV')
             ,r' $\omega_{LO}\parallel\,$ GaN',transform=lltrans,fontsize=wlabelsize,ha='left',va='top')
    plt.axhline(to_unit( hbar*wLO_perp_G,'meV'),color='k',linestyle='--',linewidth=2)
    plt.text(1,to_unit(hbar*wLO_perp_G,'meV'),
             r' $\omega_{LO}\perp$ GaN',transform=lltrans,fontsize=wlabelsize,ha='left',va='bottom')

    w_IF_TO=dc.w_IF(pol='T')[0]
    w_IF_LO=dc.w_IF(pol='L')[0]

    plt.axhline(to_unit( hbar*w_IF_TO,'meV'),color='k',linestyle=':',linewidth=2)
    plt.text(1,to_unit(hbar*w_IF_TO,'meV'),
             r' $\omega_{TO}$ IF',transform=lltrans,fontsize=wlabelsize,ha='left',va='center')
    plt.axhline(to_unit( hbar*w_IF_LO,'meV'),color='k',linestyle=':',linewidth=2)
    plt.text(1,to_unit(hbar*w_IF_LO,'meV'),
             r' $\omega_{LO}$ IF',transform=lltrans,fontsize=wlabelsize,ha='left',va='center')

    plt.xlim(0,1)
    plt.xlabel("Wavevector [1/nm]")
    plt.tight_layout()


if __name__=='__main__':
    sim=Simulation('pFET',define_mesh=define_mesh,solve_flow=solve_flow, mesh_opts={'max_dz':1*nm})
    sim.load(force=True)

    dc=sim.extras['dc']
    POP_panel(sim.extras['dc'])

    plt.figure()
    iq=20
    for i,(reg,num) in enumerate(zip(['u','IF','l'],[3,0,7])):
        plt.subplot(3,1,i+1)
        phiLO=dc.get_mode_by_name('LO'+reg,num,iq=iq)[1]
        phiTO=dc.get_mode_by_name('TO'+reg,num,iq=iq)[1]
        plt.plot(dc._keepmesh.zp,phiLO,'b',label='LO')
        plt.plot(dc._keepmesh.zp,phiTO,'r',label='TO')
        plt.title({'u':'Confined to upper layer', 'IF': 'Interface Mode', 'l': 'Confined to lower layer'}[reg])
        if np.min(phiLO)>=0 and np.min(phiTO)>=0: plt.ylim(0)
        plt.axvline(dc._keepmesh._layers[0].thickness,color='k')
        plt.xlim(0,dc._keepmesh.zp[-1])
        plt.axhline(0,color='k')
        plt.ylabel("Potential [eV]")
        plt.xlabel("Depth [nm]")
        if reg=='u':
            plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
