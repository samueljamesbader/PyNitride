from examples.pchannel_GaN_AlN_HFET.pFET_example import define_mesh as define_electrical_mesh
from pynitride.phonons import DielectricContinuum_SWH
from pynitride.reciprocal_mesh import RMesh1D
from pynitride.paramdb import to_unit, nm, meV, hbar
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

def define_mesh(sim,**kwargs):
    define_electrical_mesh(sim,**kwargs)
    sim.rmeshes['POP']=RMesh1D.regular(1/nm,100,abskshift=.001/nm)

def solve_flow(sim):
    m=sim.dmeshes['main']
    rmesh=sim.rmeshes['POP']
    dc=DielectricContinuum_SWH(m,rmesh,
        {'TOu':5,'TOIF':1,'TOl':10,'LOu':5,'LOIF':1,'LOl':10})
    dc.solve(just_energies=True)
    sim.extras['dc']=dc

def POP_panel(dc):
    plt.figure(figsize=(4,8))
    ax_meV=plt.gca()

    wLO_perp_G,wLO_para_G,wLO_perp_A,wLO_para_A,\
    wTO_perp_G,wTO_para_G,wTO_perp_A,wTO_para_A,\
    epsinf_G,epsinf_A,tw,tb=dc._params


    # Plot the solution
    plt.plot(dc.rmesh.absk,to_unit(hbar*dc.en,"meV"))

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
    from pynitride.sim import Simulation
    sim=Simulation('pFET',define_mesh=define_mesh,solve_flow=solve_flow)
    sim.load(force=True)
    POP_panel(sim.extras['dc'])
    plt.show()
