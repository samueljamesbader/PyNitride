import matplotlib.pyplot as mpl
mpl.interactive(True)
from pynitride.paramdb import nm, cm, eV
from pynitride.poissolve.mesh.structure import Mesh, EpiStack
from pynitride.poissolve.mesh.functions import PointFunction, MaterialFunction,RegionFunction,DeltaFunction

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

def gan_qwhemt(xc,xb,xw,xs,Ndef,surface='GenericMetal',snidermode=False):
    AlN="AlN"
    GaN="GaN"
    if snidermode:
        AlN="qAlN"
        GaN="qGaN"

    # Build device
    if xc==0:
        epistack=EpiStack(['barrier',AlN,xb],['well',GaN,xw],['subs',AlN,xs],surface=surface)
    else:
        epistack=EpiStack(['cap',GaN,xc],['barrier',AlN,xb],['well',GaN,xw],['subs',AlN,xs],surface=surface)
    m=Mesh(epistack,max_dz=10,refinements=[[xc+xb,.02,1.2],[xc+xb+xw,.02,1.3]])

    sm=m.submesh([xc,xc+xb+xw+xw/3])


    # No polarization charge
    m['rho_pol']=PointFunction(m,0.0)

    # Substrate impurities
    m['DeepDonorActiveConc']=RegionFunction(m,lambda name: (name=="subs")*Ndef, pos='point')
    m['DeepAcceptorActiveConc']=RegionFunction(m,lambda name: (name=="subs")*Ndef, pos='point')


    if snidermode:
        P=MaterialFunction(m,'P')
    else:
        # Hackish addition of polarization
        P=MaterialFunction(m,
                lambda mat: {
                    "GaN":5.6e-1,
                    "AlN":0.0,
                }[mat['abbrev']])
    m['rho_pol']=P.differentiate(fill_value=0.0)
    return m,sm




def gan_hemt(xc,xb,xs,Ndef,surface='GenericMetal'):
    # Build device
    if xc==0:
        epistack=EpiStack(['barrier','AlGaN',xb],['subs','AlN',xs],surface=surface)
    else:
        epistack=EpiStack(['cap','GaN',xc],['barrier','AlGaN',xb],['subs','GaN',xs],surface=surface)
    m=Mesh(epistack,max_dz=5,refinements=[[xc,.02,1.2],[xc+xb,.02,1.2]])

    # No polarization charge
    m['rho_pol']=PointFunction(m,0.0)

    # Substrate impurities
    m['DeepDonorActiveConc']=RegionFunction(m,lambda name: (name=="subs")*Ndef, pos='point')
    m['DeepAcceptorActiveConc']=RegionFunction(m,lambda name: (name=="subs")*Ndef, pos='point')


    # Hackish addition of polarization
    P=MaterialFunction(m,
                       lambda mat: {
                           "GaN":.25*5.6e-1,
                           "AlGaN":0.0,
                       }[mat['abbrev']])
    m['rho_pol']=P.differentiate(fill_value=0.0)
    return m
def super_gan_hemt(xc,xb,xs,Ndef,surface='GenericMetal'):

    # Build device
    if xc==0:
        epistack=EpiStack(['barrier','AlN',xb],['subs','AlN',xs],surface=surface)
    else:
        epistack=EpiStack(['cap','GaN',xc],['barrier','AlN',xb],['subs','GaN',xs],surface=surface)
    m=Mesh(epistack,max_dz=10,refinements=[[xc,.02,1.2],[xc+xb,.02,1.2]])

    # No polarization charge
    m['rho_pol']=PointFunction(m,0.0)

    # Substrate impurities
    m['DeepDonorActiveConc']=RegionFunction(m,lambda name: (name=="subs")*Ndef, pos='point')
    m['DeepAcceptorActiveConc']=RegionFunction(m,lambda name: (name=="subs")*Ndef, pos='point')


    # Hackish addition of polarization
    P=MaterialFunction(m,
                       lambda mat: {
                           "GaN":.25*5.6e-1,
                           "AlN":0.0,
                       }[mat['abbrev']])
    m['rho_pol']=P.differentiate(fill_value=0.0)
    return m




if __name__=='__main__':

    from pynitride.poissolve.solvers.coupled import Coupled_FD_Poisson, Coupled_Schrodinger_Poisson
    from pynitride.poissolve.visual import plot_QFV, plot_wavefunctions

    mpl.close('all')
    if False: # pn
        pn=gan_pn(xp=450*nm,xn=550*nm,Nd=1.0e18*cm**-3,Na=1.0e18*cm**-3,Ndspike=0e13*cm**-2,surface=2*eV)
        print(pn.z.shape[0])
        #pn.plot_mesh()
        def stoppah():
            return 0
            global i
            i+=1
            if i>225:
                plot_QFV(qwhemt)
                mpl.xlim(0,50)
                return 1
        Coupled_FD_Poisson(pn).solve(rise=10)
        plot_QFV(pn)
        pn.save("pn.msh")

    if True: # qwhemt
        #qwhemt,sm=gan_qwhemt(2.5*nm,4.5*nm,25*nm,500*nm,1e16*cm**-3,surface=1*eV)
        from pynitride.poissolve.materials import read_1dp_mat
        read_1dp_mat()
        qwhemt,sm=gan_qwhemt(2.5*nm,4.5*nm,25*nm,150*nm,1e17*cm**-3,surface=1*eV,snidermode=True)
        #qwhemt.plot_mesh()
        i=0
        def stoppah():
            return 0
            global i
            i+=1
            if i>1:
                plot_QFV(qwhemt)
                mpl.xlim(0,50)
                return 1
        #Coupled_FD_Poisson(qwhemt).solve(rise=100,callback=stoppah)#low_act=5,rise=200,callback=stoppah)

        #sm=qwhemt

        import time
        starttime=time.time()
        Coupled_Schrodinger_Poisson(qwhemt,carriers=['electron','hole'],schrodinger=sm).solve(rise=100)#low_act=5,rise=200,callback=stoppah)
        print("Took {:.2g}s".format(time.time()-starttime))
        plot_QFV(qwhemt)
        #plot_wavefunctions(sm)
        plot_wavefunctions(sm,bands=['h_HH'])
        print("e-sheet {:.3g}/cm^2".format(qwhemt['n'].integrate()[-1]/cm**-2))
        print("h-sheet {:.3g}/cm^2".format(qwhemt['p'].integrate()[-1]/cm**-2))
        mpl.xlim(0,50)

    mpl.interactive(False)
    mpl.show()
    #Coupled_FD_Poisson(pn).solve()
    #plot_QFV(pn)
