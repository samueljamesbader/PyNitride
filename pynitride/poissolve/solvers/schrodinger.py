import numpy as np
from pynitride.paramdb import q, hbar, kT, eV
from pynitride.poissolve.mesh.functions import ConstantFunction, MaterialFunction, PointFunction
from pynitride.poissolve.solvers.fermidirac import FermiDirac3D
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


class SchrodingerSolver():
    def __init__(self,mesh,carriers=['electron','hole']):
        r""" Solves the Schrodinger equation along *z* for the lowest eigenstates in a potential well.

        :param mesh: the :py:class:`~poissolve.mesh.structure.Mesh` on which the Schrodinger problem is defined
        :param carriers: list of carriers (elements may be 'electron' or 'hole') to quantize
            energy levels beyond those solved for.)
        """
        self._mesh=m=mesh
        self._dopants=FermiDirac3D.identifydopants(mesh)
        self._Nc,self._Nv=FermiDirac3D.effective_dos_3d(mesh)
        self._cDE,self._vDE=FermiDirac3D.band_edge_shifts(mesh)

        self._props={c:{} for c in carriers}
        for carrier,v in self._props.items():
            bands=mesh._layers[0].material['bands',carrier].keys()
            for i,b in enumerate(bands):
                v[b]={}
                v[b]['T']=self.z_kinetic_term(m,MaterialFunction(m,['bands',carrier,b,'mzs']))
                v[b]['mxys']=MaterialFunction(m,['bands',carrier,b,'mxys'],pos='point')
                v[b]['g']=mesh._layers[0].material['bands',carrier][b]['g'] # can't vary spatially
                v[b]['DE']=MaterialFunction(m,['bands',carrier,b,'DE'],pos='point')
            m[{'electron':'Ec_eff','hole':'Ev_eff'}[carrier]]=PointFunction(m,empty=(len(bands),))

        for k in ['n','p','nderiv','pderiv']:
            if k not in m:
                m[k]=PointFunction(m,empty=())

    @staticmethod
    def z_kinetic_term(mesh,mz=None):
        r""" Generates the symmetrized *z* kinetic energy term for use in a Schrodinger solution.

        This tridiagonal matrix is the discrete, symmetrized representation of
        :math:`\frac{-\hbar^2}{2m_z}\frac{\partial^2\psi}{\partial z^2}`.
        See :ref:`Solving the Schrodinger Equation <solve_schrodinger__1d>` for details on the discrete form.


        :param mesh: the :py:class:`~poissolve.mesh.structure.Mesh` on which the Schrodinger problem is defined
        :param mz: MidFunction of the effective mass along *z*
        :return: the *z* kinetic term as a sparse (CSC) matrix
        """
        diagonal=(hbar**2/(mz*mesh._dz)).to_point_function(interp='unweighted')/mesh._dzp
        offdiagonal=-(hbar**2/(2*mz*mesh._dz *np.sqrt(mesh._dzp[:-1]*mesh._dzp[1:])))
        T=diags([offdiagonal,diagonal,offdiagonal],[-1,0,1],format='csc')
        return T

    @staticmethod
    def lateral_kinetic_term(mesh,kperp,mxy):
        r""" Generates the lateral kinetic term for use in a Schrodinger solution.

        This diagonal matrix is the discrete version  of :math:`\frac{\hbar^2k_\perp^2}{2m}`.
        See :ref:`Solving the Schrodinger Equation <solve_schrodinger>` for details on the discrete form.

        :param mesh: the Mesh on which the Schrodinger problem is defined
        :param kperp: norm of the lateral wavevector
        :param mxy: MidFunction of the lateral effective mass
        :return: the lateral kinetic term as a sparse (CSC) matrix
        """
        diagonal=(hbar**2*kperp**2/(2*mxy)).to_point_function(interp='unweighted')
        T=diags(diagonal,format='csc')
        return T

    @staticmethod
    def solve_schrodinger_problem(mesh,z_kinetic_term,potential,lateral_kinetic_term=0,
                                  num_eigenvalues=8,psi_out=None):
        if not psi_out: psi_out=PointFunction(mesh,empty=(num_eigenvalues,))

        H=z_kinetic_term+diags(potential)+lateral_kinetic_term
        energies,eigenvectors=eigsh(H,k=num_eigenvalues,sigma=np.min(potential))
        psi_out[:,:]=(1/np.sqrt(mesh._dzp))*eigenvectors.T

        return energies, psi_out

    @staticmethod
    def carrier_density(psi,g,mxys,eta,summed=True):
        return (g/(2*np.pi)*kT/hbar**2)* \
              np.sum(mxys*(psi**2*(np.log(1+np.exp(eta)))).T,axis=1)


    # (kT)**-1 * d(carrier_density)/d(eta)
    @staticmethod
    def carrier_density_deriv(psi,g,mxys,eta):
        return (-g/(2*np.pi)/hbar**2)* \
               np.sum(mxys*(psi**2*(1+np.exp(-eta))**-1).T,axis=1)

    def solve(self,activation=1, eff_mass_average=True):
        m=self._mesh
        EF=m['EF']

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
            m[{'electron':'n','hole':'p'}[carrier]]=conc
            m[{'electron':'nderiv','hole':'pderiv'}[carrier]]=deriv
        carriers=self._props.keys()
        if 'hole' not in carriers:
            m['p']=0
            m['pderiv']=0
        if 'electron' not in carriers:
            m['n']=0
            m['nderiv']=0
        for k,v in zip(['n','p','nderiv','pderiv'],
                FermiDirac3D.carrier_density(EF,
                    Ec=m['Ec_eff'] if 'electron' in carriers else m['Ec'],
                    Ev=m['Ev_eff'] if 'hole' in carriers else m['Ev'],
                    Nc=self._Nc,Nv=self._Nv,
                    conduction_band_shifts=0 if 'electron' in carriers else self._cDE,
                    valence_band_shifts=0 if 'hole' in carriers else self._vDE,
                    compute_derivs=True)):
            m[k]+=v

        m['Ndp'],m['Nam'],m['Ndpderiv'],m['Namderiv']= \
            FermiDirac3D.ionized_donor_density(m,EF,m['Ec'],m['Ev'],self._dopants,compute_derivs=True)

        if activation!=1:
            for k in ['n','p','nderiv','pderiv','Ndp','Nam','Ndpderiv','Namderiv']:
                m[k]*=activation

        m['rho']=activation*m['rho_pol']+q*(m['p']+m['Ndp']-m['n']-m['Nam'])
        m['rhoderiv']= q*(m['pderiv']+m['Ndpderiv']-m['nderiv']-m['Namderiv'])

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
    m['DEc']=MaterialFunction(m,['bands','DEc']).to_point_function(interp='z')
    m['Eg']=MaterialFunction(m,['bands','Eg']).to_point_function(interp='z')
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

    SchrodingerSolver(sm,carriers=['electron','hole']).solve()

    from pynitride.poissolve.visual import plot_wavefunctions
    mpl.figure()
    plot_wavefunctions(sm,['h_HH'])
    mpl.show()
