import numpy as np

from pynitride.poissolve.solvers import pmdb, SchrodingerSolver

pmdb.make_accessible(globals(), ["k", "hbar", "e", "eV"]);q=e;kT= k * 300
from pynitride import ConstantFunction

if __name__=='__main__':
    from pynitride.paramdb import MV, cm
    from pynitride.poissolve.devices import gan_qwhemt
    from pynitride.poissolve.mesh.functions import MaterialFunction, PointFunction
    import matplotlib.pyplot as mpl
    mpl.interactive(False)
    xc=2
    xb=5
    xw=20
    xs=300
    F=2.2*MV/cm
    m,sm=gan_qwhemt(xc,xb,xw,xs,1e16*cm**-3,surface='GenericMetal')
    z=m.z
    m['kT']=ConstantFunction(m,0)
    m['DEc']=MaterialFunction(m,['electron','DEc']).to_point_function(interp='z')
    m['Eg']=MaterialFunction(m,['Eg']).to_point_function(interp='z')
    m['mqV']=PointFunction(m,np.choose(1*(z>xc)+1*(z>xc+xb)+1*(z>xc+xb+xw),
                                       [F*(z-xc),
                                        0*z,
                                        F*(z-(xc+xb)),
                                        F*xw+0*z]))-.6*eV
    m['Ec']=m['mqV']+m['DEc']
    m['Ev']=m['Ec']-m['Eg']
    m['EF']=ConstantFunction(m,0)
    #sm=m.submesh([5,30])
    #sm.plot_mesh()
    #sm.plot_function('Ec')

    SchrodingerSolver(sm, carriers=['electron', 'hole']).solve()

    from pynitride.poissolve.visual import plot_wavefunctions
    mpl.figure()
    plot_wavefunctions(sm,['h_HH'])
    mpl.show()
