from os.path import expanduser, join
from pynitride import ROOT_DIR, MaterialFunction
from pynitride.paramdb import ParamDB, Value
from pynitride.poissolve.snider import import_1dp_input, import_1dp_output, convert_1dpmat_to_PyNitride
from pynitride.poissolve.solvers import Coupled_Schrodinger_Poisson, KPSolver
from pynitride.poissolve.visual import plot_carrierFV, plot_wavefunctions

if __name__=="__main__": pass
    #pytest.main(args=[__file__])

cm,nm=ParamDB().get_constants("cm,nm")

stud=None

def doit():
    global stud
    from pynitride.poissolve.devices import gan_pqwhemt
    m,sm=gan_pqwhemt(10,500,5e16/cm**3)
    csp=Coupled_Schrodinger_Poisson(m,schrodinger=sm)
    csp.solve(low_act=3,rise=40)
    kp=KPSolver(sm)
    energies,psi=kp.solve()
    stud=energies
    #csp=Coupled_FD_Poisson(m)
    #csp.solve(callback=callback,rise=40)
    return m,sm,csp

if __name__=='__main__':
    #import cProfile
    #cProfile.run("doit()",'crestats.txt')
    m,sm,csp=doit()
    import matplotlib.pyplot as mpl
    mpl.interactive(True)
    plot_carrierFV(m)
    #plot_wavefunctions(sm,bands=['e_'])
    #plot_wavefunctions(sm,bands=['e_Gamma'])

