import numbers

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from pynitride.poissolve.maths import tdma, fd12, fd12p, idd,iddd
from pynitride import ParamDB, MaterialFunction, PointFunction, ConstantFunction, MidFunction, SubMesh
from collections import OrderedDict
from operator import mul
from functools import reduce, lru_cache

pmdb=ParamDB(units='neu')
k,hbar,q,m_e=pmdb.quantity("k,hbar,e,m_e")

class SchrodingerSolver():
    def __init__(self,mesh,carriers=['electron','hole']):
        r""" Solves the Schrodinger equation along *z* for the lowest eigenstates in a potential well.

        :param mesh: the :py:class:`~poissolve.mesh.structure.Mesh` on which the Schrodinger problem is defined
        :param carriers: list of carriers (elements may be 'electron' or 'hole') to quantize
       """
        self._mesh=m=mesh
        self._dopants=FermiDirac3D.identifydopants(mesh)
        self._Nc,self._Nv=FermiDirac3D.effective_dos_3d(mesh)
        self._cDE,self._vDE=FermiDirac3D.band_edge_shifts(mesh)

        self._props={c:{} for c in carriers}
        for carrier,v in self._props.items():
            bands=mesh._layers[0].material[carrier,"band"]
            for i,b in enumerate(bands):
                v[b]={}
                v[b]['T']=self.z_kinetic_term(m,MaterialFunction(m,[carrier,b,'mzs']))
                v[b]['mxys']=MaterialFunction(m,[carrier,b,'mxys'],pos='point')
                v[b]['g']=mesh._layers[0].material[[carrier,b,'g']] # can't vary spatially
                v[b]['DE']=MaterialFunction(m,[carrier,b,'DE'],pos='point')
            m[{'electron':'Ec_eff','hole':'Ev_eff'}[carrier]]=PointFunction(m,empty=(len(bands),))

        for k in ['n','p','nderiv','pderiv']:
            if k not in m:
                m[k]=PointFunction(m,empty=())

    #def break_hamiltonian(self):

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
        diagonal=(hbar**2/(mz*mesh._dzp)).to_point_function(interp='unweighted')/mesh._dzm
        offdiagonal=-(hbar**2/(2*mz*mesh._dzp *np.sqrt(mesh._dzm[:-1]*mesh._dzm[1:])))
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
        psi_out[:,:]=(1/np.sqrt(mesh._dzm))*eigenvectors.T

        return energies, psi_out

    @staticmethod
    def carrier_density(psi,g,mxys,eta,kT,summed=True):
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
        kT=k*m.pmdb['T']

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
                conc+=self.carrier_density(Psi,bandparms['g'],meff,eta,kT)
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
        for key,v in zip(['n','p','nderiv','pderiv'],
                         FermiDirac3D.carrier_density(EF,
                                                      Ec=m['Ec_eff'] if 'electron' in carriers else m['Ec'],
                                                      Ev=m['Ev_eff'] if 'hole' in carriers else m['Ev'],
                                                      Nc=self._Nc,Nv=self._Nv,kT=kT,
                                                      conduction_band_shifts=0 if 'electron' in carriers else self._cDE,
                                                      valence_band_shifts=0 if 'hole' in carriers else self._vDE,
                                                      compute_derivs=True)):
            m[key]+=v

        m['Ndp'],m['Nam'],m['Ndpderiv'],m['Namderiv']= \
            FermiDirac3D.ionized_donor_density(m,EF,m['Ec'],m['Ev'],kT,self._dopants,compute_derivs=True)

        if activation!=1:
            for key in ['n','p','nderiv','pderiv','Ndp','Nam','Ndpderiv','Namderiv']:
                m[key]*=activation

        m['rho']=activation*m['rho_pol']+q*(m['p']+m['Ndp']-m['n']-m['Nam'])
        m['rhoderiv']= q*(m['pderiv']+m['Ndpderiv']-m['nderiv']-m['Namderiv'])

class KPSolver():
    def __init__(self,mesh):
        r""" Solves the Multiband kp Schrodinger equation along *z* for the lowest eigenstates in a potential well.

        :param mesh: the :py:class:`~poissolve.mesh.structure.Mesh` on which the Schrodinger problem is defined
        :param carriers: list of carriers (elements may be 'electron' or 'hole') to quantize
       """
        self._mesh=m=mesh
        self._dopants=FermiDirac3D.identifydopants(mesh)
        for k in ['n','p','nderiv','pderiv']:
            if k not in m:
                m[k]=PointFunction(m,empty=())

        # See examples GenerateHwurt
        Hwurt=np.array([
           [ 'kx*L1*kx+ky*M1*ky+kz*M2*kz+kx*U*kx+ky*U*ky+kz*U*kz+Delta1+l1*exx+m1*eyy+m2*ezz',
            'kx*N1p*ky+ky*N1m*kx++-i*Delta2+n1*exy',
            'kx*N2p*kz+kz*N2m*kx+++n2*exz', '++', '++', '+Delta3+'],
           ['ky*N1p*kx+kx*N1m*ky++i*Delta2+n1*exy',
            'kx*M1*kx+ky*L1*ky+kz*M2*kz+kx*U*kx+ky*U*ky+kz*U*kz+Delta1+m1*exx+l1*eyy+m2*ezz',
            'ky*N2p*kz+kz*N2m*ky+++n2*eyz', '++', '++', '+-i*Delta3+'],
           ['kz*N2p*kx+kx*N2m*kz+++n2*exz', 'kz*N2p*ky+ky*N2m*kz+++n2*eyz',
            'kx*M3*kx+ky*M3*ky+kz*L2*kz+kx*U*kx+ky*U*ky+kz*U*kz++m3*exx+m3*eyy+l2*ezz',
            '+-1*Delta3+', '+i*Delta3+', '++'],
           ['++', '++', '+-1*Delta3+',
            'kx*L1*kx+ky*M1*ky+kz*M2*kz+kx*U*kx+ky*U*ky+kz*U*kz+Delta1+l1*exx+m1*eyy+m2*ezz',
            'kx*N1p*ky+ky*N1m*kx++i*Delta2+n1*exy',
            'kx*N2p*kz+kz*N2m*kx+++n2*exz'],
           ['++', '++', '+-i*Delta3+', 'ky*N1p*kx+kx*N1m*ky++-i*Delta2+n1*exy',
            'kx*M1*kx+ky*L1*ky+kz*M2*kz+kx*U*kx+ky*U*ky+kz*U*kz+Delta1+m1*exx+l1*eyy+m2*ezz',
            'ky*N2p*kz+kz*N2m*ky+++n2*eyz'],
           ['+Delta3+', '+i*Delta3+', '++', 'kz*N2p*kx+kx*N2m*kz+++n2*exz',
            'kz*N2p*ky+ky*N2m*kz+++n2*eyz',
            'kx*M3*kx+ky*M3*ky+kz*L2*kz+kx*U*kx+ky*U*ky+kz*U*kz++m3*exx+m3*eyy+l2*ezz']],
          dtype='<U78')
        self._C2w,self._Clw,self._Crw,self._C0w=KPSolver.break_hamiltonian(Hwurt)
        a0=MaterialFunction(mesh,'conditions=relaxed.lattice.a')
        #c0=MaterialFunction(mesh,'conditions=relaxed.lattice.c')
        C13=MaterialFunction(mesh,'stiffness.C13')
        C33=MaterialFunction(mesh,'stiffness.C33')
        a=a0[-1]
        #c=c0[-1]
        exx=eyy=(a-a0)/a0
        ezz=-2*C13/C33*exx
        A1=MaterialFunction(mesh,'kp.A1')
        A2=MaterialFunction(mesh,'kp.A2')
        A3=MaterialFunction(mesh,'kp.A3')
        A4=MaterialFunction(mesh,'kp.A4')
        A5=MaterialFunction(mesh,'kp.A5')
        A6=MaterialFunction(mesh,'kp.A6')
        D1=MaterialFunction(mesh,'kp.D1')
        D2=MaterialFunction(mesh,'kp.D2')
        D3=MaterialFunction(mesh,'kp.D3')
        D4=MaterialFunction(mesh,'kp.D4')
        D5=MaterialFunction(mesh,'kp.D5')
        D6=MaterialFunction(mesh,'kp.D6')
        mesh['Delta1']=Delta1=MaterialFunction(mesh,'kp.DeltaCR')
        mesh['Delta2']=Delta2=Delta3=1/3*MaterialFunction(mesh,'kp.DeltaSO')
        U=hbar**2/(2*m_e)
        Atwid=A2+A4-U
        Ahat=A1+A3-U
        L1=A5+Atwid
        L2=A1-U
        M1=-A5+Atwid
        M2=Ahat
        M3=A2-U
        N1p=3*A5-Atwid
        N1m=-A5+Atwid
        N2p=np.sqrt(2)*A6-Ahat
        N2m=Ahat
        l1=D2+D4+D5
        l2=D1
        m1=D2+D4-D5
        m2=D1+D3
        m3=D2
        n1=-2*D5
        n2=np.sqrt(2)*D6
        @lru_cache(maxsize=10000)
        def kppar_m(i):
            return {
                'L1':L1[i],
                'L2':L2[i],
                'M1':M1[i],
                'M2':M2[i],
                'M3':M3[i],
                'N1p':N1p[i],
                'N1m':N1m[i],
                'N2p':N2p[i],
                'N2m':N2m[i],
                'Delta1':Delta1[i],
                'Delta2':Delta2[i],
                'Delta3':Delta3[i],
                'l1':l1[i],
                'l2':l2[i],
                'm1':m1[i],
                'm2':m2[i],
                'm3':m3[i],
                'n1':n1[i],
                'n2':n2[i],
                'exx':exx[i],
                'eyy':eyy[i],
                'ezz':ezz[i],
                'exy':0,
                'exz':0,
                'eyz':0,
            }
        self._kppar_m=kppar_m
        @lru_cache(maxsize=10000)
        def kppar_p(i):
            return {
                'L1':L1.to_point_function()[i],
                'L2':L2.to_point_function()[i],
                'M1':M1.to_point_function()[i],
                'M2':M2.to_point_function()[i],
                'M3':M3.to_point_function()[i],
                'N1p':N1p.to_point_function()[i],
                'N1m':N1m.to_point_function()[i],
                'N2p':N2p.to_point_function()[i],
                'N2m':N2m.to_point_function()[i],
                'Delta1':Delta1.to_point_function()[i],
                'Delta2':Delta2.to_point_function()[i],
                'Delta3':Delta3.to_point_function()[i],
                'l1':l1.to_point_function()[i],
                'l2':l2.to_point_function()[i],
                'm1':m1.to_point_function()[i],
                'm2':m2.to_point_function()[i],
                'm3':m3.to_point_function()[i],
                'n1':n1.to_point_function()[i],
                'n2':n2.to_point_function()[i],
                'exx':exx.to_point_function()[i],
                'eyy':eyy.to_point_function()[i],
                'ezz':ezz.to_point_function()[i],
                'exy':0,
                'exz':0,
                'eyz':0,
            }
        self._kppar_p=kppar_p

        self._ss=SchrodingerSolver(mesh,carriers=['electron'])

    @staticmethod
    def break_hamiltonian(H):
        C2_terms=[[[] for j in range(H.shape[1])] for i in range(H.shape[0])]
        Cl_terms=[[[] for j in range(H.shape[1])] for i in range(H.shape[0])]
        Cr_terms=[[[] for j in range(H.shape[1])] for i in range(H.shape[0])]
        C0_terms=[[[] for j in range(H.shape[1])] for i in range(H.shape[0])]

        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                terms=H[i,j].split("+")
                for t in terms:
                    if t=="": continue
                    factors=t.split("*")
                    #print(factors)
                    if len(factors)>2 and factors[0]=='kz' and factors[2]=='kz':
                        C2_terms[i][j]+=[[factors[1]]]
                    elif len(factors)>2 and factors[2]=='kz':
                        Cl_terms[i][j]+=[[factors[1],factors[0]]]
                    elif len(factors)>1 and factors[0]=='kz':
                        Cr_terms[i][j]+=[[factors[1],factors[2]]]
                    else:
                        C0_terms[i][j]+=[factors]
        #return C2_terms,Cl_terms,Cr_terms,C0_terms
        #print(C0_terms[1][1])
        def get(f,**kwargs):
            if f=="U":
                return hbar**2/(2*m_e)
            if f in kwargs:
                return kwargs[f]
            elif f=='i': return 1j
            elif f=='-i': return -1j
            elif f=='1': return 1
            elif f=='-1': return -1

        @lru_cache(maxsize=64)
        def C2(**kwargs):
            return np.array( \
            [[sum(reduce(mul,(get(f,**kwargs) for f in t)) for t in e) for e in r] for r in C2_terms])
        @lru_cache(maxsize=64)
        def Cl(**kwargs):
            return np.array( \
            [[sum(reduce(mul,(get(f,**kwargs) for f in t)) for t in e) for e in r] for r in Cl_terms])
        @lru_cache(maxsize=64)
        def Cr(**kwargs):
            return np.array( \
            [[sum(reduce(mul,(get(f,**kwargs) for f in t)) for t in e) for e in r] for r in Cr_terms])
        @lru_cache(maxsize=64)
        def C0(**kwargs):
            return np.array( \
            [[sum(reduce(mul,(get(f,**kwargs) for f in t)) for t in e) for e in r] for r in C0_terms])
        return C2,Cl,Cr,C0

    #@staticmethod
    #def z_kinetic_term(mesh,mz=None):
    #    r""" Generates the *z* kinetic energy term for use in a Schrodinger KP solution.

    #    This tridiagonal matrix is the discrete, symmetrized representation of
    #    :math:`\frac{-\hbar^2}{2m_z}\frac{\partial^2\psi}{\partial z^2}`.
    #    See :ref:`Solving the Schrodinger Equation <solve_schrodinger__1d>` for details on the discrete form.


    #    :param mesh: the :py:class:`~poissolve.mesh.structure.Mesh` on which the Schrodinger problem is defined
    #    :param mz: MidFunction of the effective mass along *z*
    #    :return: the *z* kinetic term as a sparse (CSC) matrix
    #    """
    #    diagonal=(hbar**2/(mz*mesh._dzp)).to_point_function(interp='unweighted')/mesh._dzm
    #    offdiagonal=-(hbar**2/(2*mz*mesh._dzp *np.sqrt(mesh._dzm[:-1]*mesh._dzm[1:])))
    #    T=diags([offdiagonal,diagonal,offdiagonal],[-1,0,1],format='csc')
    #    return T

    @lru_cache(maxsize=500)
    def assemble(self,kx,ky):
        r""" Generates the lateral kinetic term for use in a Schrodinger solution."""
        #diagonal=(hbar**2*kperp**2/(2*mxy)).to_point_function(interp='unweighted')
        #T=diags(diagonal,format='csc')
        #return T
        print("Assembling: kx,ky=",kx,ky)
        mesh=self._mesh
        from scipy.sparse import lil_matrix
        T=lil_matrix((6*len(mesh.zp),6*len(mesh.zp)),dtype='complex')

        # Gotta get these memoized
        print("fill")
        for i in range(len(mesh.zp)):
            T[6*i:6*i+6, 6*i:6*i+6]=\
                self._C0w(kx=kx,ky=ky,**self._kppar_p(i))\
                +(self._C2w(kx=kx,ky=ky,**self._kppar_m(i-1))/mesh._dzp[i-1]/mesh.dzm[i] if i>0 else 0)\
                +(self._C2w(kx=kx,ky=ky,**self._kppar_m(i))/mesh._dzp[i]/mesh.dzm[i] if i<len(mesh._dzp) else 0)
        for i in range(len(mesh.zm)):
            T[6*i:6*i+6, 6*i+6:6*i+12]=\
                -self._C2w(kx=kx,ky=ky,**self._kppar_m(i))/mesh._dzp[i]/np.sqrt(mesh._dzm[i]*mesh._dzm[i+1]) \
                +1j*(self._Clw(kx=kx,ky=ky,**self._kppar_p(i))+self._Crw(kx=kx,ky=ky,**self._kppar_p(i+1)))\
                    /(2*np.sqrt(mesh._dzm[i]*mesh._dzm[i+1]))
        for i in range(1,len(mesh.zm)+1):
            T[6*i:6*i+6, 6*i-6:6*i]= \
                -self._C2w(kx=kx,ky=ky,**self._kppar_m(i-1))/mesh._dzp[i-1]/np.sqrt(mesh._dzm[i]*mesh._dzm[i-1]) \
                -1j*(self._Clw(kx=kx,ky=ky,**self._kppar_p(i))+self._Crw(kx=kx,ky=ky,**self._kppar_p(i-1))) \
                 /(2*np.sqrt(mesh._dzm[i]*mesh._dzm[i-1]))
        print("convert")
        return T.asformat('csr')

    def solve(self,num_eigenvalues=25,activation=1):
        assert activation==1
        self._ss.solve()
        mesh=self._mesh
        nm=1
        kT=k*mesh.pmdb['T']
        ens=[]
        uens=[]
        normsqs=[]
        weights=[]
        kt=np.linspace(0,1/nm,10)
        for kxi in kt:
            psi_out=PointFunction(self._mesh,empty=(num_eigenvalues,6))
            kyi=0
            Hw=-self.assemble(kxi,kyi)
            shift=np.maximum(np.maximum((mesh['Delta1'] + mesh['Delta2']).to_point_function(),(mesh['Delta1']-mesh['Delta2']).to_point_function()),0)
            pot=np.reshape(np.reshape(np.tile(-self._mesh['Ev']+shift,6),(6,len(self._mesh.zp))).transpose(),(6*len(self._mesh.zp)))
            H=Hw+diags(pot)
            print('about to eigsh ',np.min(pot))
            energies,eigenvectors=eigsh(H,k=num_eigenvalues,sigma=np.min(pot),which='LM')
            indarr=np.argsort(energies)
            energies=-energies[indarr]
            eigenvectors=eigenvectors[:,indarr]
            # first axis = position, second axis = eigenvector, third axis = component
            eigenvectors=np.rollaxis(np.reshape(eigenvectors,(len(mesh._zp),6,num_eigenvalues)),2,1)
            print('done eigsh')
            # first axis = position, second axis = eigenvector, value = normsq
            normsq=(np.sum(abs(eigenvectors)**2,axis=2).T/self._mesh._dzm).T
            # first axis = component, second axis = eigenvector, value = normsq
            weight=np.sum((abs(eigenvectors)**2).T,axis=2)
            #psi_out[:,:]=(1/np.sqrt(self._mesh._dzm))*eigenvectors.T
            ens+=[sorted(energies)]
            uens+=[energies]
            #psis+=[psi_out]
            normsqs+=[normsq]
            weights+=[weight]
        uens,normsqs,weights=np.array(uens), np.array(normsqs), np.array(weights)
        E_i=ConstantFunction(mesh,np.tile(np.atleast_3d(uens),len(mesh.zp)))
        print( "WHY AM I ONLY LOOKING AT K=0 ^^^")
        eta=np.rollaxis((E_i-mesh['EF'])/kT,2,1)
        mesh['p']=np.sum(1/(2*np.pi)*np.trapz(kt*(normsqs/(1+np.exp(-eta))).T,x=kt),axis=0)
        mesh['pderiv']=np.sum(1/(2*np.pi*kT)*np.trapz(kt*(normsqs*(np.exp(eta))/(1+np.exp(eta))**2).T,x=kt),axis=0)
        print("not blending")

        m=mesh
        EF=m['EF']
        m['Ndp'],m['Nam'],m['Ndpderiv'],m['Namderiv']= \
            FermiDirac3D.ionized_donor_density(m,EF,m['Ec'],m['Ev'],kT,self._dopants,compute_derivs=True)
        m['rho']=m['rho_pol']+q*(m['p']+m['Ndp']-m['n']-m['Nam'])
        m['rhoderiv']= q*(m['pderiv']+m['Ndpderiv']-m['nderiv']-m['Namderiv'])

        return kt,uens,normsqs,weights


#    @staticmethod
#    def solve_schrodinger_problem(mesh,z_kinetic_term,potential,lateral_kinetic_term=0,
#                                  num_eigenvalues=8,psi_out=None):
#        if not psi_out: psi_out=PointFunction(mesh,empty=(num_eigenvalues,))
#
#        H=z_kinetic_term+diags(potential)+lateral_kinetic_term
#        energies,eigenvectors=eigsh(H,k=num_eigenvalues,sigma=np.min(potential))
#        psi_out[:,:]=(1/np.sqrt(mesh._dzm))*eigenvectors.T
#
#        return energies, psi_out
#
#    @staticmethod
#    def carrier_density(psi,g,mxys,eta,kT,summed=True):
#        return (g/(2*np.pi)*kT/hbar**2)* \
#              np.sum(mxys*(psi**2*(np.log(1+np.exp(eta)))).T,axis=1)
#
#
#    # (kT)**-1 * d(carrier_density)/d(eta)
#    @staticmethod
#    def carrier_density_deriv(psi,g,mxys,eta):
#        return (-g/(2*np.pi)/hbar**2)* \
#               np.sum(mxys*(psi**2*(1+np.exp(-eta))**-1).T,axis=1)
#
#    def solve(self,activation=1, eff_mass_average=True):
#        m=self._mesh
#        EF=m['EF']
#        kT=k*m.pmdb['T']
#
#        for carrier,bands in self._props.items():
#            electron,hole=(carrier=="electron"),(carrier=="hole")
#            conc=0
#            deriv=0
#            for i,(b, bandparms) in enumerate(bands.items()):
#                abbrev="_"+carrier[0]+"_"+b
#
#                if electron: U=(m['Ec']+bandparms['DE'])
#                elif hole:   U=-(m['Ev']-bandparms['DE'])
#
#                E,Psi=self.solve_schrodinger_problem(m,bandparms['T'],U)
#                if hole: E=-E
#
#                m['Psi'+abbrev]=Psi
#                m['Energies'+abbrev]=E_i=ConstantFunction(m,E)
#
#                assert eff_mass_average, "Solving with full k-integral is not supported."
#                meff=1/((Psi**2/bandparms['mxys']).integrate()[:,-1])
#
#                if electron: eta=(m['EF']-E_i)/kT
#                elif hole:   eta=(E_i-m['EF'])/kT
#                conc+=self.carrier_density(Psi,bandparms['g'],meff,eta,kT)
#                deriv+=self.carrier_density_deriv(Psi,bandparms['g'],meff,eta)
#
#                if electron: np.maximum(E_i[-1],m['Ec'],out=m['Ec_eff'][i,:])
#                elif hole:   np.minimum(E_i[-1],m['Ev'],out=m['Ev_eff'][i,:])
#            if hole: deriv=-deriv
#            m[{'electron':'n','hole':'p'}[carrier]]=conc
#            m[{'electron':'nderiv','hole':'pderiv'}[carrier]]=deriv
#        carriers=self._props.keys()
#        if 'hole' not in carriers:
#            m['p']=0
#            m['pderiv']=0
#        if 'electron' not in carriers:
#            m['n']=0
#            m['nderiv']=0
#        for key,v in zip(['n','p','nderiv','pderiv'],
#                FermiDirac3D.carrier_density(EF,
#                    Ec=m['Ec_eff'] if 'electron' in carriers else m['Ec'],
#                    Ev=m['Ev_eff'] if 'hole' in carriers else m['Ev'],
#                    Nc=self._Nc,Nv=self._Nv,kT=kT,
#                    conduction_band_shifts=0 if 'electron' in carriers else self._cDE,
#                    valence_band_shifts=0 if 'hole' in carriers else self._vDE,
#                    compute_derivs=True)):
#            m[key]+=v
#
#        m['Ndp'],m['Nam'],m['Ndpderiv'],m['Namderiv']= \
#            FermiDirac3D.ionized_donor_density(m,EF,m['Ec'],m['Ev'],kT,self._dopants,compute_derivs=True)
#
#        if activation!=1:
#            for key in ['n','p','nderiv','pderiv','Ndp','Nam','Ndpderiv','Namderiv']:
#                m[key]*=activation
#
#        m['rho']=activation*m['rho_pol']+q*(m['p']+m['Ndp']-m['n']-m['Nam'])
#        m['rhoderiv']= q*(m['pderiv']+m['Ndpderiv']-m['nderiv']-m['Namderiv'])


class PoissonSolver():
    r""" Solves the Poisson equation on a mesh.

    The boundary conditions assumed are that the potential is zero at the first mesh point, and the electric field
    goes to zero at the last mesh point (or, to be more precise, at the next midpoint after the last meshpoint).
    Two solve functions are available: :py:func:`~pynitride.poissolve.poisson.solve` and
    :py:func:`~pynitride.poissolve.poisson.PoissonSolver.isolve`.  The former is a full, direct solution, which can be obtained
    directly from charge integration.  The latter is a Newton-method solver appropriate for self-consistent
    iteration with a charge solver such as FermiDirac3D or Schrodinger.

    :param mesh: the :py:class:`~pynitride.poissolve.mesh.Mesh` on which to perform the solve
    """
    def __init__(self, mesh):
        self._mesh = mesh

        if isinstance(mesh._layers.surface,numbers.Real):
            self._phib=mesh._layers.surface
        else:
            self._phib=mesh._layers[0].material('surface={}.electronbarrier'.format(mesh._layers.surface))

        # ARE THESE NECESSARY
        eps=self._eps=mesh['eps']= MaterialFunction(mesh, 'dielectric.eps')
        mesh['mqV']= PointFunction(mesh, 0.0)
        mesh['DEc']= MaterialFunction(mesh, 'electron.DEc', pos='point')
        self._Eg= MaterialFunction(mesh,'Eg', pos='point')


        self._left=np.empty(len(mesh.zp))
        self._right=np.empty(len(mesh.zp))
        self._left[1:]=eps/(mesh.dzp * mesh.dzm[1:])
        self._right[:-1]=eps/(mesh.dzp * mesh._dzm[:-1])
        self._center=-MidFunction(mesh,eps/mesh.dzp).to_point_function(interp='unweighted')/mesh.dzm
        self._center[-1]=self._center[-2]

        self._left[:2]=0
        self._right[0]=0
        self._right[-1:]=0
        self._center[0]=1
        self._center[1:-1]*=2

        self._mqV_temp = PointFunction(mesh)  # temp

    def solve(self):
        m=self._mesh
        qrho=q*m['rho']
        qrho[0]=0
        m['mqV']=tdma(self._left,self._center,self._right,qrho)
        self._update_others()
        mqV=m['mqV']
        m['E']=mqV.differentiate()
        m['D']=MidFunction(m,self._eps*m['E'])
        m['arho2']=m['D'].differentiate()

    def _update_others(self):
        m=self._mesh
        m['Ec']=m['mqV']+m['EF'][0]+self._phib+m['DEc']-m['DEc'][0]
        m['Ev']=m['Ec']-self._Eg

    def isolve(self,visual=False):
        m=self._mesh
        qrho=q*m['rho']
        qrho[0]=0

        # left right and center are for +d^2/dx^2, ie center is negative
        # isolve uses -d^2/dx^2, ie center (without rhoderiv) is positive

        diag = -self._center + q * self._mesh['rhoderiv']
        diag[0] -= q * m['rhoderiv'][0]
        diag[-1] -= q * m['rhoderiv'][-1]

        a=-self._left
        b=diag
        c=-self._right

        d= (q*m['arho2'] - qrho)
        d[0]=0
        d[-1]=-m['D'][-1]/m._dzm[-1]

        # What I had after redoing Neumann at bottom
        d[-1]=-m['rho'][-1]-m['D'][-1]/m._dzm[-1]

        ## Trying to just fix the last point of D at zero
        d[-1]=-m['D'][-1]/m._dzm[-1]



        import numpy as np
        if visual:
            import matplotlib.pyplot as mpl
            mpl.figure()
            mpl.subplot(311)
            mpl.plot(m.zp, qrho - q * self._rhoprev)
            #print(np.max(np.abs(qrho-q*self._rhoprev)))
            mpl.title('rho- rhoprev')
            mpl.subplot(312)
            mpl.plot(m.zp, m['rhoderiv'])
            mpl.title('rhoderiv')
            mpl.tight_layout()

        dqmV=tdma(a,b,c,d)
        #print(dqmV[-15:])
        m['mqV']+=dqmV
        self._update_others()
        self._rhoprev=m['rho'].copy()


        mqV=m['mqV']
        m['E']=mqV.differentiate()
        m['D']=MidFunction(m,self._eps*m['E'])
        m['arho2']=m['D'].differentiate()


        return np.max(np.abs(dqmV))#np.sum(np.abs(dqmV))/np.sum(np.abs(m['mqV']))


class FermiDirac3D():
    def __init__(self,mesh):
        self._mesh=mesh

        self._Nc,self._Nv=self.effective_dos_3d(mesh)
        self._cDE,self._vDE=self.band_edge_shifts(mesh)
        self._dopants=self.identifydopants(mesh)


    @staticmethod
    def identifydopants(mesh):
        dopants={'Donor':{},'Acceptor':{}}
        for d in [k[:-10] for k in mesh if k.endswith("ActiveConc")]:
            types=set(t for t in (l.material('dopant='+d+'.type',default=None) for l in mesh._layers) if t is not None)
            if len(types)>1: raise Exception(
                "Can't have one dopant be acceptor in one material and donor in another.  "\
                "You'll have to use two separate dopant names.  Sorry. ")
            if len(types)==1:
                dopants[list(types)[0]][d]={'conc':mesh[d+'ActiveConc']}
            else:
                #print("No materials include {} as a dopant.".format(d))
                pass
        for doptype in dopants.keys():
            for d,v in dopants[doptype].items():
                v['E']=MaterialFunction(mesh,d+'.E',pos='point')
                v['g']=MaterialFunction(mesh,d+'.g',pos='point')
                if doptype=="donor":
                    assert np.all(v['g'][~np.isnan(v['g'])]==2)
                if doptype=="acceptor":
                    assert np.all(v['g'][~np.isnan(v['g'])]==4)
        mesh['Nd']=np.sum(d['conc'] for d in dopants['Donor'].values())\
            if len(dopants['Donor']) else ConstantFunction(mesh,0)
        mesh['Na']=np.sum(d['conc'] for d in dopants['Acceptor'].values()) \
            if len(dopants['Acceptor']) else ConstantFunction(mesh,0)
        return dopants

    @staticmethod
    def effective_dos_3d(mesh):
        kT=k*mesh.pmdb['T']
        # We'll have to confirm these formulae (factors of 2?) later...
        Nc=MaterialFunction(mesh,pos='point',prop=lambda mat:
            mat['electron.band.g']*(mat['electron.band.mdos']*kT/(2*np.pi*hbar**2))**(3/2))
        Nv=MaterialFunction(mesh,pos='point',prop=lambda mat:
            mat['hole.band.g']*(mat['hole.band.mdos']*kT/(2*np.pi*hbar**2))**(3/2))
        mesh['Nv']=Nv
        return Nc,Nv

    @staticmethod
    def band_edge_shifts(mesh):
        cDE=MaterialFunction(mesh,pos='point',prop=lambda mat: mat['electron.band.DE'])
        vDE=MaterialFunction(mesh,pos='point',prop=lambda mat: mat['hole.band.DE'])
        return cDE,vDE

    @staticmethod
    def carrier_density(EF,Ec,Ev,Nc,Nv,kT,conduction_band_shifts=None,valence_band_shifts=None,compute_derivs=True):

        # Can save a copy and loop by checking if shifts are zero
        Ec_eff=Ec if conduction_band_shifts is None else Ec+conduction_band_shifts
        Ev_eff=Ev if valence_band_shifts is None else Ev-valence_band_shifts

        # Can save time by moving the summation inside the FD integral?
        n=np.sum(Nc*fd12((EF-Ec_eff)/kT),axis=0)
        p=np.sum(Nv*fd12((Ev_eff-EF)/kT),axis=0)

        if compute_derivs:
            nderiv=np.sum(-(Nc/kT)*fd12p((EF-Ec_eff)/kT),axis=0)
            pderiv=np.sum((Nv/kT)*fd12p((Ev_eff-EF)/kT),axis=0)
            return n,p,nderiv,pderiv
        else:
            return n,p

    @staticmethod
    def ionized_donor_density(mesh,EF,Ec,Ev,kT,dopants,compute_derivs=True):

        # Tiwari Compound Semiconductor Devices pg31-32

        #Ndp=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']*(1/(1+d['g']*np.exp((EF-Ec+d["E"])/kT)))
        #                                             for d in dopants['Donor'].values())))
        #Nam=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']*(1/(1+d['g']*np.exp((Ev+d["E"]-EF)/kT)))
        #                                             for d in dopants['Acceptor'].values())))

        Ndp=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']*idd((EF-Ec+d["E"])/kT,2)
           for d in dopants['Donor'].values())))
        Nam=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']*idd((Ev+d["E"]-EF)/kT,4)
           for d in dopants['Acceptor'].values())))

        #if compute_derivs:
        #    Ndpderiv=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']* \
        #                                                      (d['g']/kT)*np.exp((EF-Ec+d["E"])/kT)/(1+d['g']*np.exp((EF-Ec+d["E"])/kT))**2
        #                                                      for d in dopants['Donor'].values())))
        #    Namderiv=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']* \
        #                                                      (-d['g']/kT)*np.exp((Ev+d["E"]-EF)/kT)/(1+d['g']*np.exp((Ev+d["E"]-EF)/kT))**2
        #                                                      for d in dopants['Acceptor'].values())))
        #    return Ndp,Nam,Ndpderiv,Namderiv
        #else:
        #    return Ndp,Nam


        if compute_derivs:
            Ndpderiv=PointFunction(mesh,np.nan_to_num(np.sum( d['conc']/kT*iddd((EF-Ec+d["E"])/kT,2)\
                for d in dopants['Donor'].values())))
            Namderiv=PointFunction(mesh,np.nan_to_num(np.sum( -d['conc']/kT*iddd((Ev+d["E"]-EF)/kT,4)\
                for d in dopants['Acceptor'].values())))
            return Ndp,Nam,Ndpderiv,Namderiv
        else:
            return Ndp,Nam

    def solve(self,activation=1, quantum_band_shift=False):
        m=self._mesh
        EF=m['EF']
        Ec=m['Ec']
        Ev=m['Ev']
        kT=k*m.pmdb['T']

        m['Ndp'],m['Nam'],m['Ndpderiv'],m['Namderiv']=\
            self.ionized_donor_density(m,EF,Ec,Ev,kT,self._dopants,compute_derivs=True)

        if quantum_band_shift:
            m['n'],m['p'],m['nderiv'],m['pderiv']=self.carrier_density(EF,Ec,Ev,self._Nc,self._Nv,kT,
                conduction_band_shifts=self._cDE,valence_band_shifts=self._vDE,compute_derivs=True)
        else:
            m['n'],m['p'],m['nderiv'],m['pderiv']=self.carrier_density(EF,Ec,Ev,self._Nc,self._Nv,kT,
               conduction_band_shifts=None,valence_band_shifts=None,compute_derivs=True)

        if activation!=1:
            for key in ['n','p','nderiv','pderiv','Ndp','Nam','Ndpderiv','Namderiv']:
                m[key]*=activation

        m['rho']=activation*m['rho_pol']+q*(m['p']+m['Ndp']-m['n']-m['Nam'])
        m['rhoderiv']= q*(m['pderiv']+m['Ndpderiv']-m['nderiv']-m['Namderiv'])

class Linear_Fermi():

    def __init__(self,mesh,contacts={'gate':0,'subs':-1}):
        self._mesh=mesh
        interfaces=[(0,None)]+mesh._interfacesp+[((len(mesh.zp)-1),None)]
        self._contacts=OrderedDict(sorted([(k,interfaces[v][0]) for k,v in contacts.items()],key=lambda x:x[1]))
        mesh['EF']=PointFunction(mesh)

    def solve(self,**voltages):
        lefts=list(self._contacts.items())[:-1]
        rights=list(self._contacts.items())[1:]
        for (clname,cl),(crname,cr) in zip(lefts,rights):
            l=-voltages.get(clname,0)
            r=-voltages.get(crname,0)
            self._mesh['EF'][cl:(cr+1)]=(self._mesh.zp[cl:(cr+1)]-self._mesh.zp[cl])/(self._mesh.zp[cr]-self._mesh.zp[cl])*(r-l)+l

class Coupled_FD_Poisson():

    def __init__(self,mesh):
        m=mesh
        # Set some stuff
        m['EF']=PointFunction(m,0.0)
        if 'rho_pol' not in m:
            m['rho_pol']=PointFunction(m,0.0)
        if 'rho' not in m:
            m['rho']=PointFunction(m,0.0)
        if 'arho2' not in m:
            m['arho2']=PointFunction(m,0.0) # is this necessary?

        # Prep solvers_old
        self._ps=PoissonSolver(m)
        self._fd=FermiDirac3D(m)


    def solve(self, low_act=4, rise=500, tol=1e-8, max_iter=100, callback=lambda *args: None):
        self._ps.solve()
        if callback(): return
        for activation in np.logspace(-low_act,-0.,rise):
            self._fd.solve(activation=activation)
            err=self._ps.isolve(visual=False)
            #print(err)
            if callback(): return
        #print("Rose")
        for i in range(max_iter):
            self._fd.solve(activation=1)
            err=self._ps.isolve(visual=False)
            if callback(): return
            #print(err)
            if err<tol:
                print("Success (max err={:.2g}) after {:d} refinement iterations".format(err,i-1))
                break
        assert err<tol, "Stopped because reached max_iter with err ({:.2g}) > tol ({:.2g}).".format(err,tol)


class Coupled_Schrodinger_Poisson():

    def __init__(self,mesh, carriers=['electron','hole'],schrodinger=None):
        m=mesh
        # Set some stuff
        m['EF']=PointFunction(m,0.0)
        if 'rho_pol' not in m:
            m['rho_pol']=PointFunction(m,0.0)
        if 'rho' not in m:
            m['rho']=PointFunction(m,0.0)
        if 'arho2' not in m:
            m['arho2']=PointFunction(m,0.0) # is this necessary?
        self._classical_charge_solvers=[FermiDirac3D(m)]


        schrofull=(schrodinger is None)
        if schrofull: schrodinger=m
        self._quantum_charge_solvers=[SchrodingerSolver(schrodinger,carriers=carriers)]
        if not schrofull:
            if schrodinger._slicep.start is not None and schrodinger._slicep.start>0:
                fd_sm=SubMesh(m,None,schrodinger._slicep.start)
                fd=FermiDirac3D(fd_sm)
                self._quantum_charge_solvers+=[fd]
            if schrodinger._slicep.stop is not None and schrodinger._slicep.stop<len(m.zp):
                fd_sm=SubMesh(m,schrodinger._slicep.stop,None)
                fd=FermiDirac3D(fd_sm)
                self._quantum_charge_solvers+=[fd]

        # Prep solvers_old
        self._ps=PoissonSolver(m)

    def solve(self, low_act=4, rise=500, tol=1e-10, max_iter=100, callback=lambda *args: None, skip_classical=False):
        self._ps.solve()
        #if callback(): return
        if not skip_classical:
            for activation in np.logspace(-low_act,-0.,rise):
                #self._fd.solve(activation=activation)
                for s in self._classical_charge_solvers: s.solve(activation)
                err=self._ps.isolve(visual=False)
                if callback(): return
            for i in range(max_iter):
                for s in self._classical_charge_solvers: s.solve(activation=1)
                err=self._ps.isolve(visual=False)
                if callback(): return
                if err<tol:
                    print("Semi-classical success (max err={:.2g}) after {:d} refinement iterations".format(err,i-1))
                    break
            assert err<tol, "Stopped because reached max_iter with err ({:.2g}) > tol ({:.2g}).".format(err,tol)
        for i in range(max_iter):
            for s in self._quantum_charge_solvers:
                if isinstance(s,SchrodingerSolver):
                    s.solve(activation=1)
                elif isinstance(s,FermiDirac3D):
                    s.solve(activation=1,quantum_band_shift=True)
            err=self._ps.isolve(visual=False)
            if callback(): return
            if err<tol:
                print("Full success (max err={:.2g}) after {:d} refinement iterations".format(err,i-1))
                break
        assert err<tol, "Stopped because reached max_iter with err ({:.2g}) > tol ({:.2g}).".format(err,tol)




























    ##########################################################################
        ######################################################################

class Coupled_KP_Poisson():

    def __init__(self,mesh, carriers=['electron','hole'],schrodinger=None):
        m=mesh
        # Set some stuff
        m['EF']=PointFunction(m,0.0)
        if 'rho_pol' not in m:
            m['rho_pol']=PointFunction(m,0.0)
        if 'rho' not in m:
            m['rho']=PointFunction(m,0.0)
        if 'arho2' not in m:
            m['arho2']=PointFunction(m,0.0) # is this necessary?
        self._classical_charge_solvers=[FermiDirac3D(m)]


        schrofull=(schrodinger is None)
        if schrofull: schrodinger=m
        self._quantum_charge_solvers=[SchrodingerSolver(schrodinger)]
        if not schrofull:
            if schrodinger._slicep.start is not None and schrodinger._slicep.start>0:
                fd_sm=SubMesh(m,None,schrodinger._slicep.start)
                fd=FermiDirac3D(fd_sm)
                self._quantum_charge_solvers+=[fd]
            if schrodinger._slicep.stop is not None and schrodinger._slicep.stop<len(m.zp):
                fd_sm=SubMesh(m,schrodinger._slicep.stop,None)
                fd=FermiDirac3D(fd_sm)
                self._quantum_charge_solvers+=[fd]

        schrofull=(schrodinger is None)
        if schrofull: schrodinger=m
        self._quantum_charge_solvers2=[KPSolver(schrodinger)]
        if not schrofull:
            if schrodinger._slicep.start is not None and schrodinger._slicep.start>0:
                fd_sm=SubMesh(m,None,schrodinger._slicep.start)
                fd=FermiDirac3D(fd_sm)
                self._quantum_charge_solvers2+=[fd]
            if schrodinger._slicep.stop is not None and schrodinger._slicep.stop<len(m.zp):
                fd_sm=SubMesh(m,schrodinger._slicep.stop,None)
                fd=FermiDirac3D(fd_sm)
                self._quantum_charge_solvers2+=[fd]

        # Prep solvers_old
        self._ps=PoissonSolver(m)

    def solve(self, low_act=4, rise=500, tol=1e-10, max_iter=100, callback=lambda *args: None, skip_classical=False):
        self._ps.solve()
        #if callback(): return
        if not skip_classical:
            for activation in np.logspace(-low_act,-0.,rise):
                #self._fd.solve(activation=activation)
                for s in self._classical_charge_solvers: s.solve(activation)
                err=self._ps.isolve(visual=False)
                if callback(): return
            for i in range(max_iter):
                for s in self._classical_charge_solvers: s.solve(activation=1)
                err=self._ps.isolve(visual=False)
                if callback(): return
                if err<tol:
                    print("Semi-classical success (max err={:.2g}) after {:d} refinement iterations".format(err,i-1))
                    break
            assert err<tol, "Stopped because reached max_iter with err ({:.2g}) > tol ({:.2g}).".format(err,tol)
        for i in range(max_iter):
            for s in self._quantum_charge_solvers:
                if isinstance(s,SchrodingerSolver):
                    s.solve(activation=1)
                elif isinstance(s,FermiDirac3D):
                    s.solve(activation=1,quantum_band_shift=True)
            err=self._ps.isolve(visual=False)
            if callback(): return
            if err<tol:
                print("Full success (max err={:.2g}) after {:d} refinement iterations".format(err,i-1))
                break
        for i in range(max_iter):
            for s in self._quantum_charge_solvers2:
                if isinstance(s,KPSolver):
                    s.solve(activation=1)
                elif isinstance(s,FermiDirac3D):
                    s.solve(activation=1,quantum_band_shift=True)
            err=self._ps.isolve(visual=False)
            if callback(): return
            if err<tol:
                print("Full success (max err={:.2g}) after {:d} refinement iterations".format(err,i-1))
                break
        assert err<tol, "Stopped because reached max_iter with err ({:.2g}) > tol ({:.2g}).".format(err,tol)
