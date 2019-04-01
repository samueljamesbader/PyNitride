# AlN/GaN/AlN Quantum Well HEMT with GaN cap

import matplotlib.pyplot as plt
from pynitride import Mesh, NodFunction, MidFunction, MaterialBlock, UniformLayer
from pynitride.physics.material import AlGaN
from pynitride import to_unit, nm, eV, m_e, cm
from pynitride import Schrodinger, Semiclassical, MultibandKP
from pynitride import PoissonSolver, Linear_Fermi, SelfConsistentLoop
from pynitride import ConstantT, Pseudomorphic
from time import time
import numpy as np
from operator import itemgetter
import os.path
from scipy.signal import savgol_filter


def make_mesh(name):
    sim={}

    # Set up the mesh
    cap_t =1.5*nm
    barr_t=4.5*nm
    well_t=30*nm
    buff_t=200*nm
    m=Mesh([
        MaterialBlock("epi",AlGaN(),[
            UniformLayer("cap"   ,   cap_t, x=0, DeepDonorDonorConc=5e16/cm**3),
            UniformLayer("barr"  ,  barr_t, x=1, DeepDonorDonorConc=5e16/cm**3),
            UniformLayer("well"  ,  well_t, x=0, DeepDonorDonorConc=5e16/cm**3),
            UniformLayer("buffer",  buff_t, x=1, DeepDonorDonorConc=5e16/cm**3),
        ])],
        max_dz=5*nm,
        refinements=[['barr/cap',.01*nm,1.4],['well/buffer',.01*nm,1.4]],uniform=False,boundary=[.7*eV,"thick"])

    print("Mesh points: ", m.Nn)
    schro,semi=m.submesh_cover([cap_t+barr_t+well_t+5*nm])

    return {'main':m,'schro':schro,'semi':semi,'name':name}

def solve(meshes,to_Va,numV,outdir=""):
    m,schro,semi,name= \
        itemgetter('main','schro','semi','name')(meshes)
    lf=Linear_Fermi(m,contacts={'gate':0,'subs':-1,'S2DEG':2,'S2DHG':3})
    lf.solve(gate=0,subs=0,S2DEG=0)
    ConstantT(m)
    Pseudomorphic(m)
    ps=PoissonSolver(m)

    solv_semi =Semiclassical(schro)
    solv_schro=Schrodinger(schro)
    scl=SelfConsistentLoop(
        fieldsolvers =[PoissonSolver(m)],
        carriermodels=[solv_semi,
                       Semiclassical(semi)])
    scl.ramp_epsfactor(start=1e4, max_iter=20, dlefmin=.005, tol=1e-5)
    #scl.swap_carrier_model(remove=solv_semi,add=solv_schro)
    #scl.loop(tol=1e-5)

    if to_Va!=0:
        Va=np.linspace(0,to_Va,numV)
        rho=[]
        p=[]
        n=[]
        rrats=[]
        for v in Va:
            print("Applied voltage: {:.4f} V".format(v))
            rrat=100*(1+np.exp((v-2)/.3))/(1+np.exp((v-1)/.3))
            #rrat=10000
            print("Rrat: {:.2g}".format(rrat))

            ps.update_bands_to_potential(m,0)
            m['p']=0
            m['n']=0
            lf.solve(gate=v,S2DEG=v/(1+rrat),S2DHG=0,subs=0)
            #scl.swap_carrier_model(remove=solv_schro,add=solv_semi)
            scl.ramp_epsfactor(start=1e4, max_iter=20, dlefmin=.005, tol=1e-7)
            #scl.swap_carrier_model(remove=solv_semi,add=solv_schro)
            #scl.loop(tol=1e-5)
            rho+=[float(m.rho.integrate(definite=True))]
            p+=[float(m.p.integrate(definite=True))]
            n+=[float(m.n.integrate(definite=True))]
            rrats+=[rrat]
            if np.isclose(v,int(v)):
                m.save(name+"_Va_"+str(int(v)))

        np.savez(name+"_CV_to_{}".format(to_Va),Va=Va,rho=rho,p=p,n=n,rrat=rrats)

def load(to_Va,numV,outdir="",force=False,name='QWHEMT'):
    meshes=make_mesh(name)
    m,schro,semi=itemgetter('main','schro','semi')(meshes)

    if not force:
        # Try to load from file
        try:
            m.read(os.path.join(outdir,name+"_Va_"+str(int(to_Va))+".npz"))
            PoissonSolver.update_bands_to_potential(m)
            print("Loading previous run from ",outdir)
            return meshes
        except Exception as e:
            print(e)
            pass

    # Otherwise redo the solve
    print("Previous run not loaded, so running solve")
    solve(meshes,to_Va=to_Va,numV=numV,outdir=outdir)
    return meshes

if __name__=="__main__":
    name='QWHEMT-leaky'

    meshes=load(name=name,to_Va=0,numV=0)
    m,schro,semi=itemgetter('main','schro','semi')(meshes)
    plt.figure()
    plt.plot(m.Ec,m.zm,color='b')
    plt.plot(m.Ev,m.zm,color='g')
    plt.plot(m.EF, m.zn, color='r')
    scale=10
    plt.fill_betweenx(m.zn, -scale * m.p + m.EF, m.EF, color='g', alpha=.2)
    plt.fill_betweenx(m.zn, +scale * m.n + m.EF, m.EF, color='b', alpha=.2)
    plt.ylim(40,0)
    plt.ylabel("Depth [nm]")
    plt.xlabel("Energy [eV]")
    plt.title("Zero bias")

    #meshes=load(name=name,to_Va=-5,numV=1001,force=False)
    meshes=load(name=name,to_Va=3,numV=151,force=True)

    #Va,rho,p,n=itemgetter('Va','rho','p','n')(np.load(name+'_CV_to_-5.npz'))
    Va,rho,p,n,rrat=itemgetter('Va','rho','p','n','rrat')(np.load(name+'_CV_to_3.npz'))
    dV=Va[1]-Va[0]
    assert np.allclose(np.diff(Va),dV)
    C=-savgol_filter(rho,11,1,1)/dV

    plt.figure()
    plt.plot(Va,to_unit(C,"fF/um^2"),'r',label="Capacitance")
    #plt.ylim(0,15)
    #plt.xlim(-5,0)
    plt.xlim(0,3)
    plt.ylabel(r"Capacitance [fF/$\mathrm{\mu}$m$^2$]")
    plt.legend(loc='upper left')
    plt.xlabel("Voltage [V]")
    plt.twinx()
    plt.plot(Va,to_unit(n,"1e13/cm^2"),'b--',label="Electrons")
    plt.plot(Va,to_unit(p,"1e13/cm^2"),'g--',label="Holes")
    plt.ylim(0,6)
    plt.ylabel("Charge [10$^{13}$/cm$^2$]")
    plt.legend(loc='upper right')
    plt.figure()
    plt.plot(Va,rrat)


    m,schro,semi=itemgetter('main','schro','semi')(meshes)
    plt.figure()
    plt.plot(m.Ec,m.zm,color='b')
    plt.plot(m.Ev,m.zm,color='g')
    plt.plot(m.EF, m.zn, color='r')
    scale=10
    plt.fill_betweenx(m.zn, -scale * m.p + m.EF, m.EF, color='g', alpha=.2)
    plt.fill_betweenx(m.zn, +scale * m.n + m.EF, m.EF, color='b', alpha=.2)
    plt.ylim(40,0)
    plt.ylabel("Depth [nm]")
    plt.xlabel("Energy [eV]")
    plt.title("Negative Bias")


    plt.show()
