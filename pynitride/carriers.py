import numpy as np
from pynitride.mesh import PointFunction, MidFunction, Function, ConstantFunction
from pynitride.paramdb import pmdb, k, pi, hbar, m_e, nm
from pynitride.maths import fd12, fd12p, assemble6x6
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.sparse import lil_matrix
from pynitride.visual import log, sublog


class CarrierModel():
    """ Superclass for all carrier models.

    A carrier model implements a :func:`solve()` function and a :func:`repopulate` function, which together define how
    to get from previously-solved bands to the induced carrier densities.
    """

    def solve(self):
        """ Performs any prep work needed (e.g. solving for wavefunctions) before the mesh can be populated.

        This method generally does the expensive eigenvalue calculations, whereas :func:`repopulate` simply takes these
        results and fills in the levels.

        Precondition: the mesh already includes the bands "Ec" and "Ev" (e.g. provided by :class:`Poisson`).

        Post-condition: One can now call repopulate()
        """
        raise NotImplementedError

    def repopulate(self):
        """ Populates the mesh with carriers.

        This method should be quick because :func:`solve` has already done the heavy lifting.
        So this can be called inside tighter loops where the Fermi level changes but bands do not.

        Precondition: self.solve() has already been called to set up any necessary resources (eg wavefunctions)
        and the mesh includes the Fermi level "EF" (e.g. provided by :class:`Equilibrium`).

        Post-condition: the mesh will have the electron and hole densities "n" and "p",
        as well as their approximate functional derivatives "nderiv" and "pderiv" with respect to the bands
        (ie :math:`n' = \delta n / \delta [-q \phi],\ p' = \delta p / \delta [-q \phi]`)
        """
        raise NotImplementedError

class Semiclassical(CarrierModel):
    """ Implements a semiclassical carrier model.

    Carriers at each point are populated locally according to band alignment as if that point was the bulk.
    See :func:`solve` for details and see :ref:`Carrier Models <carriers_semiclassical>` for the physics.

    Arguments:
        mesh (:class:`.mesh.Mesh`): the Mesh to populate

        carriers (list of strings): which carriers to populate,
        eg ``["electron"]``, ``["hole"]``, or ``["electron","hole"]``
    """
    def __init__(self,mesh,carriers=['electron','hole']):
        m=self._mesh=mesh
        self._carriers=carriers

        # Compute the effective density of states
        if 'electron' in carriers:
            m['Nc']= m.eg * (m.medos*k*m.T/(2*pi*hbar**2))**(3/2)
        if 'hole' in carriers:
            m['Nv']= m.hg * (m.mhdos*k*m.T/(2*pi*hbar**2))**(3/2)

    def solve(self):
        """ Does nothing."""
        pass

    def repopulate(self,Ec_eff=None, Ev_eff=None):
        """ Populate carriers semi-classically into the mesh.  See :func:`CarrierModel.solve` for conditions,
        and see :ref:`Carrier Models <carriers_semiclassical>` for the physics.

        Accepts effective condition and valence bands as optional arguments to use in place of the mesh Ec and Ev.
        For use as a stand-alone model, these should not be supplied, but they are convenient when this model is called
        by other models which may populate some levels quantum mechanically and then use this function to fill in the
        rest of the band semiclassically.

        Arguments:
            Ec_eff (Point :class:`.mesh.Function`): Effective conduction band level, optional.

            Ev_eff (Point :class:`.mesh.Function`): Effective valence band level, optional.
        """
        m=self._mesh

        # Use effective band edges if provided, otherwise take from mesh
        Ec_eff = Ec_eff if Ec_eff is not None else m.Ec
        Ev_eff = Ev_eff if Ev_eff is not None else m.Ev

        # Compute the carrier density and its derivative
        if 'electron' in self._carriers:
            eta=((m.EF - Ec_eff).tmf() - m.cDE) / (k * m.T)
            m['n']=     np.sum(m.Nc * fd12(eta), axis=0).tpf()
            m['nderiv']=np.sum(-(m.Nc/(k*m.T)) * fd12p(eta), axis=0).tpf()
        if 'hole' in self._carriers:
            eta=((Ev_eff - m.EF).tmf() - m.vDE) / (k * m.T)
            m['p']=     np.sum(m.Nv * fd12(eta), axis=0).tpf()
            m['pderiv']=np.sum((m.Nv/(k*m.T)) * fd12p(eta), axis=0).tpf()

class Schrodinger(CarrierModel):
    """ Implements a Schrodinger envelope function carrier model.

    Carriers at each point are populated according to band alignment as if that point was the bulk.
    See :func:`solve` for details and see :ref:`Carrier Models <carriers_schrodinger>` for the physics.

    Arguments:
        mesh (:class:`.mesh.Mesh`): the Mesh to populate

        carriers (list of strings): which carriers to populate, eg ``["electron"]``, ``["hole"]``, or ``["electron","hole"]``

        num_eigenvalues (int): how many eigenvalues to solve for and occupy quantum mechanically. [Default = 8]

        blend (bool): whether to add in semiclassically-occupied carriers at energies beyond the solved for levels. [Default = True]

        transverse (str): "parabolic" [Default] or "full k-space", see physics section for details
    """
    def __init__(self,mesh,carriers=['electron','hole'],
                 num_eigenvalues=8,blend=True,transverse="parabolic",
                 boundary=["Dirichlet","Dirichlet"]):

        # Store values
        m=self._mesh=mesh
        self._blend=blend
        if transverse != "parabolic":
            raise NotImplementedError
        self._transverse="parabolic"
        self._carriers=carriers
        self._neig=num_eigenvalues
        self._boundary=boundary

        if 'electron' in carriers:
            self._ezkinetic=[]
            self._elkineticfactor=[]
            self._ebands=m.mez.shape[0]
            m['een']=PointFunction(m,empty=(self._ebands,self._neig))
            m['epsi']=PointFunction(m,empty=(self._ebands,self._neig))
            for b in range(self._ebands):
                print("WHAT IS THIS BC")
                diagonal=(hbar**2/(m.mez[b,:]*m._dzp)).tpf(interp='unweighted')/m._dzm+m.cDE[b,:].tpf()
                offdiagonal=-(hbar**2/(2*m.mez[b,:]*m._dzp *np.sqrt(m._dzm[:-1]*m._dzm[1:])))
                self._ezkinetic+=[
                    diags([offdiagonal,diagonal,offdiagonal],[-1,0,1],format='csc')]
                self._elkineticfactor+=[
                    diags((hbar**2/(2*m.mexy[b,:])).tpf(interp='unweighted'),format='csc')]   #*kperp**2


            if blend:
                self._sce=Semiclassical(m,carriers=['electron'])
        if 'hole' in carriers:
            self._hzkinetic=[]
            self._hlkineticfactor=[]
            self._hbands=m.mhz.shape[0]
            for b in range(self._hbands):
                diagonal=-(hbar**2/(m.mhz[b,:]*m._dzp)).tpf(interp='unweighted')/m._dzm-m.vDE[b,:].tpf()
                offdiagonal=(hbar**2/(2*m.mhz[b,:]*m._dzp *np.sqrt(m._dzm[:-1]*m._dzm[1:])))
                self._hzkinetic+=[
                    diags([offdiagonal,diagonal,offdiagonal],[-1,0,1],format='csc')]
                self._hlkineticfactor+=[
                    -diags((hbar**2/(2*m.mhxy[b,:])).tpf(interp='unweighted'),format='csc')]   #*kperp**2
            m['hen']=PointFunction(m,empty=(self._hbands,self._neig))
            m['hpsi']=PointFunction(m,empty=(self._hbands,self._neig))
            if blend:
                self._sch=Semiclassical(m,carriers=['hole'])

    def solve(self):
        kperp=0
        m=self._mesh
        log("Using averaged effective mass to parabolically populate schrodinger bands",level='TODO')
        if 'electron' in self._carriers:
            for b in range(self._ebands):
                H=self._ezkinetic[b]+diags(m.Ec)+self._elkineticfactor[b]*kperp**2
                energies,eigenvectors=eigsh(H,k=self._neig+self._blend,sigma=np.min(m.Ec))
                m['een'][b,:,:]=np.atleast_2d(energies[:-1]).T
                m['epsi'][b,:,:]=(1/np.sqrt(m._dzm))*eigenvectors[:,:-1].T
            eta=np.rollaxis((m.EF-m.een).tmf()/(k*m.T),1,0)
            psisq=np.rollaxis(abs(m.epsi.tmf())**2,1,0)
            mmean=np.atleast_3d((m.mexy*psisq).integrate(definite=True))
            if self._blend: self._sce.solve(Ec_eff=np.maximum(energies[-1],m.Ec))
            else: m['n']=m['nderiv']=0
            m['n']+=np.sum(np.sum(
                (m.eg*mmean*k*m.T)/(2*pi*hbar**2)*psisq*np.log(1+np.exp(eta)),
                axis=0),axis=0).tpf()
            m['nderiv']+=np.sum(np.sum(
                -(m.eg*mmean)/(2*pi*hbar**2)*psisq*np.exp(eta)/(1+np.exp(eta)),
                axis=0),axis=0).tpf()
        if 'hole' in self._carriers:
            for b in range(self._hbands):
                H=self._hzkinetic[b]+diags(m.Ev)+self._hlkineticfactor[b]*kperp**2
                energies,eigenvectors=eigsh(-H,k=self._neig+self._blend,sigma=np.min(-m.Ev))
                print(energies)
                m['hen'][b,:,:]=np.atleast_2d(-energies[:-1]).T
                m['hpsi'][b,:,:]=(1/np.sqrt(m._dzm))*eigenvectors[:,:-1].T
            eta=np.rollaxis((m.hen-m.EF).tmf()/(k*m.T),1,0)
            psisq=np.rollaxis(abs(m.hpsi.tmf())**2,1,0)
            mmean=np.atleast_3d((m.mhxy*psisq).integrate(definite=True))
            if self._blend: self._sch.solve(Ev_eff=np.minimum(-energies[-1],m.Ev))
            else: m['n']=m['nderiv']=0
            m['p']+=np.sum(np.sum(
                (m.hg*mmean*k*m.T)/(2*pi*hbar**2)*psisq*np.log(1+np.exp(eta)),
                axis=0),axis=0).tpf()
            m['pderiv']+=np.sum(np.sum(
                (m.hg*mmean)/(2*pi*hbar**2)*psisq*np.exp(eta)/(1+np.exp(eta)),
                axis=0),axis=0).tpf()

class MultibandKP(CarrierModel):
    def __init__(self,mesh,num_eigenvalues=20,ktmax=2/nm,num_kpoints=25):
        m=self._mesh=mesh
        self._neig=num_eigenvalues
        m['kppsi']=PointFunction(m,dtype='complex',empty=(num_kpoints,num_eigenvalues,6,))
        m['kpen']=PointFunction(m,dtype='float',empty=(num_kpoints,num_eigenvalues,))
        self._normsqs=PointFunction(m,dtype='float',empty=(num_kpoints,num_eigenvalues))

        print("Assembling k.p matrices ...")
        self._kt=np.linspace(0,ktmax,num_kpoints)
        Cmats=m._matsys.kp_Cmats(m,kx=self._kt,ky=0*self._kt)
        self._H=[assemble6x6(C0,Cl,Cr,C2,m._dzm,m._dzp,periodic=False) for kx,[C0,Cl,Cr,C2] in zip(self._kt,Cmats)]
        print("Done assembly.")

    def solve(self):
        log("MBKP Solve",level="debug")
        m=self._mesh
        kT=k*m.T
        kt=self._kt
        pot=-np.reshape(np.reshape(np.tile(m.Ev+m.EvOffset.tpf(),6),(6,len(m.zp))).transpose(),(6*len(m.zp)))
        #print('about to eigsh ',np.min(pot))
        for i,(kx,H) in enumerate(zip(kt,self._H)):
            Htot=-H+diags(pot)
            energies,eigenvectors=eigsh(Htot,k=self._neig,sigma=np.min(pot),which='LM')

            # Sort by energy
            indarr=np.argsort(energies)
            #print("kx ",kx," en ",energies[indarr][:10])
            # kt, eig, z
            m['kpen'][i,:,:]=-np.atleast_2d(energies[indarr]).T
            # kt, eig, comp, z
            m['kppsi'][i,:,:,:]=np.swapaxes(np.reshape(
                    eigenvectors[:,indarr],
                    (len(m._zp),6,self._neig)),0,2)\
                /np.sqrt(m._dzm)
            # kt, eig, z
            self._normsqs[i,:,:]=np.sum(abs(m.kppsi[i])**2,axis=1)

        # kt, eig, z
        eta=(m.kpen-m.EF)/kT.tpf()

        m['p']=np.sum(1/(2*np.pi)*np.trapz(kt*(self._normsqs/(1+np.exp(-eta))).T,x=kt),axis=1)
        m['pderiv']=np.sum(1/(2*np.pi*kT.tpf())*np.trapz(kt*(self._normsqs*(np.exp(-eta))/(1+np.exp(-eta))**2).T,x=kt).T,axis=0)
        log("not blending",level="TODO")
        #m['hen']=uens
        #m['hpsi']=eigenvectors.T/np.sqrt(self._mesh.dzm)
        #return kt,uens,normsqs,weights


if __name__=="__main__":
    from pynitride.mesh import UniformLayer, Mesh, PointFunction
    from pynitride.material import AlGaN
    #m=Mesh([UniformLayer("Main", 100, x=0)],AlGaN(),max_dz=.1)
    m=Mesh([UniformLayer("l", 25, x=.75),UniformLayer("m", 4, x=0),UniformLayer("r", 25, x=.75)],AlGaN(),max_dz=.1,subs={'x':0})
    #m.plot_mesh()

    m['EF']= PointFunction(m,1.7)
    def initialize_potential(m):
        m['phi']=PointFunction(m,empty=())
        m['Ec']= PointFunction(m,empty=())
        m['Ev']= PointFunction(m,empty=())
        update_potential(m,0)

    def update_potential(m,phi):
        m['phi']=phi
        m['Ev']=-m.phi+(m.DE-m['E0-Ev']).tpf()
        m['Ec']=-m.phi+(m.DE+m['Ec-E0']).tpf()

    initialize_potential(m)

    import matplotlib.pyplot as plt
    from pynitride.paramdb import cm

    #Semiclassical(m).solve()
    #plt.figure()
    #plt.plot(m.zp,m.Ec,linewidth=2)
    #plt.plot(m.zp,m.Ev,linewidth=2)
    #plt.plot(m.zp,m.EF,'--',linewidth=2)
    #plt.twinx()
    #plt.plot(m.zp,m.n *cm**3)
    #plt.plot(m.zp,m.p *cm**3)

    #Schrodinger(m).solve()
    #plt.figure()
    #plt.plot(m.zp,m.Ec,linewidth=2)
    #plt.plot(m.zp,m.Ev,linewidth=2)
    #plt.plot(m.zp,m.EF,'--',linewidth=2)
    ##plt.plot(m.zp,m.een[0,:,:].T,'--')
    ##plt.plot(m.zp,m.epsi[0,:3,:].T,'--')
    #plt.plot(m.zp,m.hpsi[:3,0,:].T,'--')
    #plt.twinx()
    #plt.plot(m.zp,m.n *cm**3)
    #plt.plot(m.zp,m.p *cm**3)
    ##plt.yscale('log')

    kt,uens,normsqs,weights=\
    MultibandKP(m).solve()
    print("Done solve")
    #plt.figure()
    #plt.plot(m.zp,m.Ec,linewidth=2)
    #plt.plot(m.zp,m.Ev,linewidth=2)
    #plt.plot(m.zp,m.EF,'--',linewidth=2)
    ##plt.plot(m.zp,m.een[0,:,:].T,'--')
    ##plt.plot(m.zp,m.epsi[0,:3,:].T,'--')
    #plt.plot(m.zp,m.hpsi[:3,0,:].T,'--')
    #plt.twinx()
    #plt.plot(m.zp,m.n *cm**3)
    #plt.plot(m.zp,m.p *cm**3)
    ##plt.yscale('log')





