from poissolve.mesh.functions import MaterialFunction, ConstantFunction, PointFunction
from poissolve.constants import hbar, kT, m0, eV
from poissolve.solvers.fermidirac import FermiDirac3D
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import numpy as np

class SchrodingerSolver():
    def __init__(self,mesh,carriers=['electron','hole'],fermidirac=None):
        if fermidirac is None:
            self._fd=FermiDirac3D(mesh)
        else:
            self._fd=fermidirac
        self._mesh=m=mesh

        self._props={c:{} for c in carriers}
        for carrier,v in self._props.items():
            bands=mesh._layers[0].material['ladder',carrier].keys()
            for b in bands:
                v[b]={}
                v[b]['T']=self.kinetic_term(m,MaterialFunction(m,['ladder',carrier,b,'mzs']))
                v[b]['mxys']=MaterialFunction(m,['ladder',carrier,b,'mxys'],pos='point')
                v[b]['g']=mesh._layers[0].material['ladder',carrier][b]['g'] # can't vary spatially
                v[b]['DE']=MaterialFunction(m,['ladder',carrier,b,'DE'],pos='point')
            m[{'electron':'Ec_eff','hole':'Ev_eff'}[carrier]]=PointFunction(m,empty=(len(bands),))

    @staticmethod
    def kinetic_term(mesh,mz):
        """ Generates the kinetic term for use in a Schrodinger solution.

        $hbar**2$

        :param mesh: the Mesh on which the Schrodinger problem is defined
        :param mz: MidFunction of the longitudinal effective mass
        :return: the kinetic term as a sparse (CSC) matrix
        """
        diagonal=(hbar**2/(mz*mesh._dz)).to_point_function(interp='unweighted')/mesh._dzp
        offdiagonal=-(hbar**2/(2*mz*mesh._dz *np.sqrt(mesh._dzp[:-1]*mesh._dzp[1:])))
        T=diags([offdiagonal,diagonal,offdiagonal],[-1,0,1],format='csc')
        return T

    @staticmethod
    def solve_schrodinger_problem(mesh,kinetic_term,potential_term,num_eigenvalues=3,
                                  psi_out=None, kperp=None, mxys=None):
        if not psi_out: psi_out=PointFunction(mesh,empty=(num_eigenvalues,))

        H=kinetic_term+diags(potential_term)
        if kperp is not None:
            H+=diags(hbar**2*kperp**2/(2*mxys))
        energies,eigenvectors=eigsh(H,k=num_eigenvalues,sigma=np.min(potential_term))

        psi_out[:,:]=(1/np.sqrt(mesh._dzp))*eigenvectors.T

        return energies, psi_out

    @staticmethod
    def carrier_density(psi,g,mxys,eta):
        return (g/(2*np.pi)*kT/hbar**2)* \
              np.sum(mxys*(psi**2*(np.log(1+np.exp(eta)))).T,axis=1)

    # (kT)**-1 * d(carrier_density)/d(eta)
    @staticmethod
    def carrier_density_deriv(psi,g,mxys,eta):
        return (-g/(2*np.pi)/hbar**2)* \
               np.sum(mxys*(psi**2*(1+np.exp(-eta)**-1)).T,axis=1)

    def solve(self,activation=1, eff_mass_average=True):
        m=self._mesh

        for carrier,bands in self._props.items():
            electron,hole=(carrier=="electron"),(carrier=="hole")
            conc=0
            deriv=0
            for i,(b, bandparms) in enumerate(bands.items()):
                abbrev="_"+carrier[0]+"_"+b

                if electron: U=(m['Ec']+bandparms['DE'])
                elif hole:   U=-(m['Ev']-bandparms['DE'])

                E,Psi=self.solve_schrodinger_problem(m,bandparms['T'],U)
                if hole: E=-E

                m['Psi'+abbrev]=Psi
                m['Energies'+abbrev]=E_i=ConstantFunction(m,E)

                assert eff_mass_average, "Solving with full k-integral is not supported."
                meff=1/((Psi**2/bandparms['mxys']).integrate()[:,-1])

                if electron: eta=(m['EF']-E_i)/kT
                elif hole:   eta=(E_i-m['EF'])/kT
                conc+=self.carrier_density(Psi,bandparms['g'],meff,eta)
                deriv+=self.carrier_density_deriv(Psi,bandparms['g'],meff,eta)

                if electron: np.maximum(E_i[-1],m['Ec'],out=m['Ec_eff'][i,:])
                elif hole:   np.minimum(E_i[-1],m['Ev'],out=m['Ev_eff'][i,:])
            if hole: deriv=-deriv


            m[{'electron':'n_quantum','hole':'p_quantum'}[carrier]]=conc*activation
            m[{'electron':'nderiv_quantum','hole':'pderiv_quantum'}[carrier]]=deriv*activation

        self._fd.solve(activation=activation,quantized_bands=self._props.keys())


if __name__=='__main__':
    from poissolve.constants import MV, cm
    from poissolve.devices import gan_qwhemt
    from poissolve.mesh.functions import MaterialFunction, PointFunction
    import matplotlib.pyplot as mpl
    mpl.interactive(False)
    xc=2
    xb=5
    xw=20
    xs=300
    F=3*MV/cm
    m=gan_qwhemt(xc,xb,xw,xs,1e16*cm**-3,surface='GenericMetal')
    z=m.z
    m['kT']=ConstantFunction(m,0)
    m['DEc']=MaterialFunction(m,'DEc').to_point_function(interp='z')
    m['Eg']=MaterialFunction(m,'Eg').to_point_function(interp='z')
    m['mqV']=PointFunction(m,np.choose(1*(z>xc)+1*(z>xc+xb)+1*(z>xc+xb+xw),
                                       [F*(z-xc),
                                        0*z,
                                        F*(z-(xc+xb)),
                                        F*xw+0*z]))-.6*eV
    m['Ec']=m['mqV']+m['DEc']
    m['Ev']=m['Ec']-m['Eg']
    m['EF']=ConstantFunction(m,0)
    sm=m.submesh([5,30])
    #sm.plot_mesh()
    #sm.plot_function('Ec')

    SchrodingerSolver(sm,carriers=['electron']).solve()

    from poissolve.visual import plot_wavefunctions
    mpl.figure()
    plot_wavefunctions(sm)
    mpl.show()
