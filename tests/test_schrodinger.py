import pytest
from pynitride.poissolve.mesh.functions import MaterialFunction, MidFunction
from pynitride import PointFunction, MidFunction, ConstantFunction, MaterialFunction
from pynitride.poissolve.solvers import SchrodingerSolver, PoissonSolver
from pynitride.paramdb import ParamDB
pmdb=ParamDB(units='neu')
#from pynitride.poissolve.devices import gan_qwhemt
import matplotlib.pyplot as mpl
import numpy as np

mpl.interactive(False)
# should be move back to top of file
if __name__=='__main__':
    pass;#pytest.main(args=[__file__,'--plots'])
    #test_schrodinger()
#from poissolve.tests.runtests import plots

#@plots
#def test_schrodinger():
#    xc=2
#    xb=5
#    xw=20
#    xs=300
#    F=3*MV/cm
#    m=gan_qwhemt(xc,xb,xw,xs,1e16*cm**-3,surface='GenericMetal')
#    z=m.z
#    m['DEc']=MaterialFunction(m,'DEc').to_point_function(interp='z')
#    m['mqV']=PointFunction(m,np.choose(1*(z>xc)+1*(z>xc+xb)+1*(z>xc+xb+xw),
#       [F*(z-xc),
#       0*z,
#       F*(z-(xc+xb)),
#       F*xw+0*z]))
#    m['Ec']=m['mqV']+100*m['DEc']
#    sm=m.submesh([5,30])
#    #sm.plot_mesh()
#    sm.plot_function('Ec')
#    mpl.show()
#    #mpl.interactive(True)
#
#    SchrodingerSolver(sm).solve()
#    assert 0

#@plots
def test_li_kuhn_1994():
    global N, E0,num
    from pynitride.poissolve.mesh.functions import MaterialFunction, PointFunction, ConstantFunction
    from pynitride.poissolve.mesh import Mesh, EpiStack
    import matplotlib.pyplot as mpl
    mpl.interactive(False)

    L=9.611
    xs=3.5*L
    V0=.4

    mpl.figure()
    for meshtype in ['uniform', 'flexible', 'refined']:
        N=[]
        E0=[]
        num=[]
        for Ntarget in np.linspace(100,3000,33):
            m=Mesh(EpiStack(['AlGaAs',xs],['GaAs',L],['AlGaAs',xs],surface='GenericMetal',pmdb=pmdb),max_dz=(2*xs+L)/Ntarget,
                   refinements=([[xs,(2*xs+L)/Ntarget/5,1.4],[xs+L,(2*xs+L)/Ntarget/5,1.4]] if meshtype=='refined' else None),uniform=(meshtype=='uniform'))
            m['rho_pol']=ConstantFunction(m,0)
            #mpl.figure()
            #m.plot_mesh()
            N+=[len(m.z)]
            z=m.z

            from pynitride.poissolve.solvers_old.poisson import PoissonSolver
            m['rho']=ConstantFunction(m,0)
            m['EF']=ConstantFunction(m,0)
            PoissonSolver(m).solve()


            m['kT']=ConstantFunction(m,0)
            m['DEc']=MaterialFunction(m,['electron','DEc']).to_point_function(interp='z')
            m['mqV']=PointFunction(m,np.choose(1*(z>xs-1e-8)+1*(z>L+xs-1e-8),
                                               [V0-m['DEc'][0]+0*z,
                                                0*z,
                                                V0-m['DEc'][0]+0*z]))
            m['Ec']=m['mqV']+m['DEc']



            #m.plot_function('Ec')
            import time
            start=time.time()
            SchrodingerSolver(m,carriers=['electron']).solve()
            print("points {} time {}".format(m.z.shape[0],time.time()-start))
            E0+=[m['Energies_e_'][0,0]]
            num+=[np.max(m._dz)/np.min(m._dz)-1]
            print('hi')
            #from poissolve.visual import plot_wavefunctions
            #mpl.figure()
            #plot_wavefunctions(m)
            #mpl.show()

        Etrue=0.035766166225935696
        mpl.plot(N,np.abs(np.array(E0)-Etrue)/Etrue,'-')
        mpl.xscale('log')
        mpl.yscale('log')
        mpl.xlim(1e2,6e4)
        mpl.ylim(1e-6,1e-1)
        mpl.xlabel('Number of mesh points')
        mpl.ylabel('Relative error')
        mpl.legend(['Uniform mesh', 'Flexible mesh', 'Refined Mesh'], loc='upper right')
        #mpl.twinx()
        #mpl.plot(N,num)
        #mpl.ylim(0,.1)
    mpl.show()
    mpl.interactive(True)

if __name__=='__main__':
    test_li_kuhn_1994()
