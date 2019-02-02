""" Tests Schrodinger equation in a GaAs/AlGaAs square quantum well

 Based on `Li 1992 <https://doi.org/10.1006/jcph.1994.1026>`_.
"""

from pynitride.mesh import Mesh, PointFunction, MidFunction, MaterialBlock, UniformLayer
from pynitride.material import MaterialSystem
from pynitride.paramdb import nm, eV, m_e, hbar
from pynitride.carriers import Schrodinger
from pynitride.solvers import PoissonSolver, Equilibrium
import numpy as np
pi=np.pi
import matplotlib.pyplot as plt
from scipy.optimize import brentq

class AlGaAs(MaterialSystem):
    def __init__(self):
        #super().__init__()

        self.name="AlGaAs"
        self._attrs={
            'medos':    self.smcls_band_params,
            'mexy':     self.smcls_band_params,
            'mez':      self.smcls_band_params,
            'cDE':      self.smcls_band_params,
            'eg':       self.smcls_band_params,

            'Eg':       self.bandedge_params,
            'DE':       self.bandedge_params,
            'Ec-E0':    self.bandedge_params,
            'E0-Ev':    self.bandedge_params,
            'eps':      lambda m,key: 0*m.zp,
        }
        self._defaults={'x': 0}
        self._updates={}
        self._dopants=[]

    def smcls_band_params(self,m,key):
        m['eg']=MidFunction(m,2)
        m['cDE']=MidFunction(m,[0])

        # Equation 27, fiddle with mu to get the effective mass exact
        alpha=.067*m_e
        mu=.083*m_e
        #mu=(.1-.067)*m_e/.401
        m['mez']=m['mexy']=m['medos']=\
            MidFunction(m,np.expand_dims(alpha+m.x*mu,0))

        return m[key]

    def bandedge_params(self,m,key):

        # Equation 26, fiddle with mu to get the band offset exact
        delta=1.424
        nu=1.247
        #nu=.4*1.25/(.401)
        m['Eg']=delta+m.x*nu

        # 80/20 split of bands
        m['DE']=(m.x*nu*.80-m.x*nu*.20)/2

        # E0 is defined as the center
        m['E0-Ev']=m.Eg/2
        m['Ec-E0']=m.Eg/2

        return m[key]

    def surface_barrier(self,m):
        return m['Ec-E0'][0]+m.DE[0]


def find_sqw_energies(L,mb,mw,DEc):
    """ Slight generalization of the solution on wikipedia https://en.wikipedia.org/wiki/Finite_potential_well
    mw/mb*u = v tan v  or mw/mb*u = -v cot v, where u^2+mb/mw*v^2=u0^2, u=a L/2, v = k L/2, E=hbar^2 k^2/2mw"""

    u0=np.sqrt(mb*L**2*DEc/(2*hbar**2))
    def left(v):
        return mw/mb*np.sqrt(u0 ** 2 - mb/mw*v ** 2)

    def right_s(v):
        return v * np.tan(v)

    def right_a(v):
        return -v / np.tan(v)

    i = 0
    v = []
    while True:
        try:
            if i * pi > np.sqrt(mw/mb)*u0: break
            lower, upper = i * pi, min((i + .5) * pi, np.sqrt(mw/mb)*u0)
            v += [brentq(lambda v: left(v) - right_s(v), lower + 1e-8, upper - 1e-8)]

            if (i + .5) * pi > np.sqrt(mw/mb)*u0: break
            lower, upper = (i + .5) * pi, min((i + 1) * pi, np.sqrt(mw/mb)*u0)
            v += [brentq(lambda v: left(v) - right_a(v), lower + 1e-8, upper - 1e-8)]
            i += 1
        except Exception as e:
            print(e)
            break
    k=2*np.array(v)/L
    E=hbar**2*k**2/(2*mw)
    return E

L=9.611*nm
mole=.401
def compare(refinement,uniform):
    # Set up the mesh
    m=Mesh([
        MaterialBlock("epi",AlGaAs(),[
            UniformLayer("l",  3*L,x=mole),
            UniformLayer("m",    L,x=   0),
            UniformLayer("r",  3*L,x=mole),
        ])],
        max_dz=10*nm,
        refinements=[['l/m',refinement,1.2],['m/r',refinement,1.2]],uniform=uniform)
    print("Mesh points: ",len(m.zp))
    imid=int(len(m.zp)/2)

    Equilibrium(m)
    PoissonSolver.update_bands_to_potential(m,0.)

    s=Schrodinger(m,carriers=['electron'],blend=False)
    s.solve()

    # Compare with analytic answers
    E=(s._een[0,:]-m.Ec[imid])
    return m.Np,E


if __name__=="__main__":
    wellmat=AlGaAs().bulk(x=0)
    barrmat=AlGaAs().bulk(x=mole)
    print(L,barrmat.mez/m_e,wellmat.mez/m_e,barrmat.Eg/2+barrmat.DE-wellmat.Eg/2-wellmat.DE)
    energies=find_sqw_energies(L,barrmat.mez,wellmat.mez,barrmat.Eg/2+barrmat.DE-wellmat.Eg/2-wellmat.DE)
    print("Analytic")
    print(energies)
    N,E=[],[]
    for refinement in [1,.5,.1,.05,.02,.01,.005,.002,.001,.0005]:
        Ni,Ei=compare(refinement=refinement,uniform=False)
        N+=[Ni];E+=[Ei[:3]];
    plt.plot(N,np.abs(np.array(E)-energies[:3])/energies[:3],color='b')

    N,E=[],[]
    for refinement in [1,.5,.1,.05,.02,.01,.005,.002,.001,.0005]:
        Ni,Ei=compare(refinement=refinement,uniform=True)
        N+=[Ni];E+=[Ei[:3]];
    plt.plot(N,np.abs(np.array(E)-energies[:3])/energies[:3],'r')

    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1e-6,1e-1)
    plt.xlim(1e2,1e5)
    plt.ylabel("Error")
    plt.xlabel("Mesh points")
    plt.text(.95,.9, 'Variable Grid',color='b',transform=plt.gca().transAxes,ha='right',va='top',fontsize=16)
    plt.text(.95,.8,'Uniform Grid', color='r',transform=plt.gca().transAxes,ha='right',va='top',fontsize=16)
    plt.show()
