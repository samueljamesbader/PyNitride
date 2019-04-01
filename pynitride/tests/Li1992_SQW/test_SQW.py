""" Tests Schrodinger equation in a GaAs/AlGaAs square quantum well

 Based on `Li 1992 <https://doi.org/10.1006/jcph.1994.1026>`_.
"""

from pynitride import Mesh, NodFunction, MidFunction, MaterialBlock, UniformLayer
from pynitride.physics.material import MaterialSystem
from pynitride import nm, eV, m_e
from pynitride import Schrodinger
from pynitride import PoissonSolver, Equilibrium
import numpy as np

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

            'DE':       self.bandedge_params,
            'Ec-E0':    self.bandedge_params,
            'E0-Ev':    self.bandedge_params,
            'eps':      lambda m,key: 0*m.zn,
        }
        self._defaults={'x': 0}
        self._updates={}
        self._dopants=[]

    def smcls_band_params(self,m,key):
        m['eg']=MidFunction(m,2)
        m['cDE']=MidFunction(m,[0])

        # Equation 27, fiddle with mu to get the effective mass exact
        alpha=.067*m_e
        #mu=.083*m_e
        mu=(.1-.067)*m_e/.401
        m['mez']=m['mexy']=m['medos']=\
            MidFunction(m,np.expand_dims(alpha+m.x*mu,0))

        return m[key]

    def bandedge_params(self,m,key):

        # Equation 26, fiddle with mu to get the band offset exact
        delta=1.424
        #nu=1.247
        nu=.4*1.25/(.401)
        m['Eg']=delta+m.x*nu

        # 80/20 split of bands
        m['DE']=(m.x*nu*.80-m.x*nu*.20)/2

        # E0 is defined as the center
        m['E0-Ev']=m.Eg/2
        m['Ec-E0']=m.Eg/2

        return m[key]

    def surface_barrier(self,m):
        return m['Ec-E0'][0]+m.DE[0]

if __name__=="__main__":
    # Set up the mesh
    L=9.611*nm
    m=Mesh([
        MaterialBlock("epi",AlGaAs(),[
            UniformLayer("l",  3*L,x=.401),
            UniformLayer("m",    L,x=   0),
            UniformLayer("r",  3*L,x=.401),
            ])],
        max_dz=10*nm,
        #refinements=[['l/m',.025*nm,1.2],['m/r',.025*nm,1.2]],uniform=True)
        refinements=[['l/m',.007*nm,1.2],['m/r',.007*nm,1.2]],uniform=False)
        #refinements=[['l/m',.14*nm,1.2],['m/r',.14*nm,1.2]],uniform=True)
        #refinements=[['l/m',.04*nm,1.2],['m/r',.04*nm,1.2]],uniform=False)
    print("Mesh points: ", len(m.zn))
    imid=int(len(m.zn) / 2)

    Equilibrium(m)
    PoissonSolver.update_bands_to_potential(m,0.)

    # Check that the potential is .4eV
    V=m.Ec[0]-m.Ec[imid]
    assert(np.isclose(V,.4*eV,atol=1e-10))
    mwell=m.mez[0,imid]
    mbarr=m.mez[0,0]

    # Check that the effective masses are .067*m_e and .100*m_e
    assert(np.isclose(mwell/m_e,.067,atol=1e-10))
    assert(np.isclose(mbarr/m_e,.100,atol=1e-10))

    s=Schrodinger(m,carriers=['electron'],blend=False)
    s.solve()

    # Compare with analytic answers
    E=(s._een[0,:]-m.Ec[imid])
    print("Electron Energies ", E)
    E0err=(E[0]/.0357664-1)*100
    E1err=(E[1]/.1423513-1)*100
    E2err=(E[2]/.3110624-1)*100
    if 1:
        import matplotlib.pyplot as plt
        plt.plot(m.zm,m.Ec,'.-b',markersize=5)
        plt.plot(m.zm,m.Ev,'g')
        plt.plot(m.zn, m.EF, 'r')
        plt.plot(m.zn, np.real(s._epsi[0, 0, :]) + np.expand_dims(s._een[0, 0], 1), '.-', color='purple', markersize=5)
        plt.plot(m.zn, np.real(s._epsi[0, 1, :]) + np.expand_dims(s._een[0, 1], 1), '.-', color='pink', markersize=5)
        plt.plot(m.zn, np.real(s._epsi[0, 2, :]) + np.expand_dims(s._een[0, 2], 1), '.-', color='black', markersize=5)
        plt.show()
    print("E0 pct err {:+.5f}%".format(E0err))
    assert(np.abs(E0err)<5e-4)
    assert(np.abs(E1err)<5e-4)
    assert(np.abs(E2err)<5e-4)

    assert np.isclose((np.conj(s._epsi[0,0,:])*s._epsi[0,0,:]).integrate(definite=True),1,atol=1e-5)
    assert np.isclose((np.conj(s._epsi[0,1,:])*s._epsi[0,0,:]).integrate(definite=True),0,atol=1e-5)
