import numpy as np
import matplotlib.pyplot as mpl
mpl.interactive(True)
from poissolve.constants import nm, cm, eV
from poissolve.solvers.poisson import PoissonSolver
from poissolve.solvers.fermidirac import FermiDirac3D
from poissolve.mesh.structure import Mesh, EpiStack
from poissolve.mesh.functions import PointFunction,MidFunction,MaterialFunction,RegionFunction,DeltaFunction

def gan_pn(xp,xn,Nd,Na,Ndspike=0,surface='GenericMetal'):

    # Build device
    epistack=EpiStack(['pGaN','GaN',xp],['nGaN','GaN',xn],surface=surface)
    m=Mesh(epistack,max_dz=1,refinements=[[xp,.2,1.3]])

    # No polarization charge
    m['rho_pol']=PointFunction(m,0.0)

    # Uniform p-n doping
    m['SiActiveConc']=RegionFunction(m,
        lambda name: (name=="nGaN")*Nd, pos='point')
    m['MgActiveConc']=RegionFunction(m,
        lambda name: (name=="pGaN")*Na, pos='point')

    # Spike doping
    m['SiActiveConc']+=Ndspike*DeltaFunction(m,xp,pos='point')

    return m

def gan_qwhemt(xc,xb,xw,xs,Ndef,surface='GenericMetal'):

    # Build device
    if xc==0:
        epistack=EpiStack(['barrier','AlN',xb],['well','GaN',xw],['subs','AlN',xs],surface=surface)
    else:
        epistack=EpiStack(['cap','GaN',xc],['barrier','AlN',xb],['well','GaN',xw],['subs','AlN',xs],surface=surface)
    m=Mesh(epistack,max_dz=10,refinements=[[xc+xb,.05,1.2],[xc+xb+xw,.05,1.3]])

    # No polarization charge
    m['rho_pol']=PointFunction(m,0.0)

    # Substrate impurities
    m['DeepDonorActiveConc']=RegionFunction(m,lambda name: (name=="subs")*Ndef, pos='point')
    m['DeepAcceptorActiveConc']=RegionFunction(m,lambda name: (name=="subs")*Ndef, pos='point')


    # Hackish addition of polarization
    P=MaterialFunction(m,
            lambda mat: {
                "GaN":5.6e-1,
                "AlN":0.0,
            }[mat['abbrev']])
    m['rho_pol']=P.differentiate(fill_value=0.0)
    return m

if __name__=='__main__':

    from poissolve.solvers.coupled import Coupled_FD_Poisson
    from poissolve.visual import plot_QFV

    mpl.close('all')
    if False: # pn
        pn=gan_pn(xp=350*nm,xn=550*nm,Nd=1.0e18*cm**-3,Na=1.0e18*cm**-3,Ndspike=0e13*cm**-2,surface=2*eV)
        pn.plot_mesh()
        Coupled_FD_Poisson(pn).solve()
        plot_QFV(pn)

    if True: # qwhemt
        qwhemt=gan_qwhemt(2.5*nm,5*nm,20*nm,500*nm,1e16*cm**-3,surface=1*eV)
        #qwhemt.plot_mesh()
        i=0
        def stoppah():
            return 0
            global i
            i+=1
            if i>225:
                plot_QFV(qwhemt)
                mpl.xlim(0,50)
                return 1
        Coupled_FD_Poisson(qwhemt).solve(low_act=5,rise=200,callback=stoppah)
        plot_QFV(qwhemt)

    mpl.interactive(False)
    mpl.show()
    #Coupled_FD_Poisson(pn).solve()
    #plot_QFV(pn)
