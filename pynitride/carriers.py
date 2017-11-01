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

    A carrier model implements a solve() function which can populate the mesh with carriers.
    """

    def solve(self):
        """ Every carrier model must implement this to populate the mesh with carriers.

        Precondition: the mesh already includes the Fermi level "EF" and bands "Ec" and "Ev".
        These may, for example, be provided by :class:`Poisson` and :class:`Equilibrium`.
        Post-condition: the mesh will have the electron and hole densities "n" and "p",
        as well as their approximate derivatives "nderiv" and "pderiv" with respect to the bands
        (ie nderiv = d [n] / d [-q phi], pderiv = d [p] / d [-q phi])
        """
        raise NotImplementedError

class Semiclassical(CarrierModel):
    def __init__(self,mesh,carriers=['electron','hole']):
        m=self._mesh=mesh
        self._carriers=carriers
        if 'electron' in carriers:
            m['Nc']= m.eg * (m.medos*k*m.T/(2*pi*hbar**2))**(3/2)
        if 'hole' in carriers:
            m['Nv']= m.hg * (m.mhdos*k*m.T/(2*pi*hbar**2))**(3/2)

    def solve(self,Ec_eff=None, Ev_eff=None):
        m=self._mesh
        Ec_eff = Ec_eff if Ec_eff is not None else m.Ec
        Ev_eff = Ev_eff if Ev_eff is not None else m.Ev

        if 'electron' in self._carriers:
            eta=((m.EF - Ec_eff).tmf() - m.cDE) / (k * m.T)
            m['n']=     np.sum(m.Nc * fd12(eta), axis=0).tpf()
            m['nderiv']=np.sum(-(m.Nc/(k*m.T)) * fd12p(eta), axis=0).tpf()
        if 'hole' in self._carriers:
            eta=((Ev_eff - m.EF).tmf() - m.vDE) / (k * m.T)
            m['p']=     np.sum(m.Nv * fd12(eta), axis=0).tpf()
            m['pderiv']=np.sum((m.Nv/(k*m.T)) * fd12p(eta), axis=0).tpf()

class Schrodinger(CarrierModel):
    def __init__(self,mesh,carriers=['electron','hole'],num_eigenvalues=8):
        m=self._mesh=mesh
        self._carriers=carriers
        self._neig=num_eigenvalues
        if 'electron' in carriers:
            self._ezkinetic=[]
            self._elkineticfactor=[]
            self._ebands=m.mez.shape[0]
            for b in range(self._ebands):
                print("WHAT IS THIS BC")
                diagonal=(hbar**2/(m.mez[b,:]*m._dzp)).tpf(interp='unweighted')/m._dzm+m.cDE[b,:].tpf()
                offdiagonal=-(hbar**2/(2*m.mez[b,:]*m._dzp *np.sqrt(m._dzm[:-1]*m._dzm[1:])))
                self._ezkinetic+=[
                    diags([offdiagonal,diagonal,offdiagonal],[-1,0,1],format='csc')]
                self._elkineticfactor+=[
                    diags((hbar**2/(2*m.mexy[b,:])).tpf(interp='unweighted'),format='csc')]   #*kperp**2
            m['een']=PointFunction(m,empty=(self._ebands,self._neig))
            m['epsi']=PointFunction(m,empty=(self._ebands,self._neig))
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

    def solve(self):
        kperp=0
        m=self._mesh
        log("Using averaged effective mass to parabolically populate schrodinger bands",level='TODO')
        if 'electron' in self._carriers:
            for b in range(self._ebands):
                H=self._ezkinetic[b]+diags(m.Ec)+self._elkineticfactor[b]*kperp**2
                energies,eigenvectors=eigsh(H,k=self._neig,sigma=np.min(m.Ec))
                m['een'][b,:,:]=np.atleast_2d(energies).T
                m['epsi'][b,:,:]=(1/np.sqrt(m._dzm))*eigenvectors.T
            eta=np.rollaxis((m.EF-m.een).tmf()/(k*m.T),1,0)
            psisq=np.rollaxis(abs(m.epsi.tmf())**2,1,0)
            mmean=np.atleast_3d((m.mexy*psisq).integrate(definite=True))
            m['n']=np.sum(np.sum(
                (m.eg*mmean*k*m.T)/(2*pi*hbar**2)*psisq*np.log(1+np.exp(eta)),
                axis=0),axis=0).tpf()
            m['nderiv']=np.sum(np.sum(
                -(m.eg*mmean)/(2*pi*hbar**2)*psisq*np.exp(eta)/(1+np.exp(eta)),
                axis=0),axis=0).tpf()
        if 'hole' in self._carriers:
            for b in range(self._hbands):
                H=self._hzkinetic[b]+diags(m.Ev)+self._hlkineticfactor[b]*kperp**2
                energies,eigenvectors=eigsh(-H,k=self._neig,sigma=np.min(-m.Ev))
                m['hen'][b,:,:]=np.atleast_2d(-energies).T
                m['hpsi'][b,:,:]=(1/np.sqrt(m._dzm))*eigenvectors.T
            eta=np.rollaxis((m.hen-m.EF).tmf()/(k*m.T),1,0)
            psisq=np.rollaxis(abs(m.hpsi.tmf())**2,1,0)
            mmean=np.atleast_3d((m.mhxy*psisq).integrate(definite=True))
            m['p']=np.sum(np.sum(
                (m.hg*mmean*k*m.T)/(2*pi*hbar**2)*psisq*np.log(1+np.exp(eta)),
                axis=0),axis=0).tpf()
            m['pderiv']=np.sum(np.sum(
                (m.hg*mmean)/(2*pi*hbar**2)*psisq*np.exp(eta)/(1+np.exp(eta)),
                axis=0),axis=0).tpf()

class MultibandKP(CarrierModel):
    def __init__(self,mesh,num_eigenvalues=20,ktmax=2/nm,num_kpoints=25):
        m=self._mesh=mesh
        self._neig=num_eigenvalues

        print("Assembling k.p matrices ...")
        self._kt=np.linspace(0,ktmax,num_kpoints)
        Cmats=m._matsys.kp_Cmats(m,kx=self._kt,ky=0*self._kt)
        self._H=[assemble6x6(C0,Cl,Cr,C2,m._dzm,m._dzp,periodic=False) for kx,[C0,Cl,Cr,C2] in zip(self._kt,Cmats)]
        print("Done assembly.")

    def solve(self):
        log("MBKP Solve",level="debug")
        m=self._mesh
        kT=k*m.T
        ens=[]
        uens=[]
        normsqs=[]
        weights=[]
        kt=self._kt
        pot=-np.reshape(np.reshape(np.tile(m.Ev+m.EvOffset.tpf(),6),(6,len(m.zp))).transpose(),(6*len(m.zp)))
        #print('about to eigsh ',np.min(pot))
        for kx,H in zip(kt,self._H):
            psi_out=PointFunction(self._mesh,empty=(self._neig,6))
            Htot=-H+diags(pot)
            energies,eigenvectors=eigsh(Htot,k=self._neig,sigma=np.min(pot),which='LM')
            indarr=np.argsort(energies)
            energies=-energies[indarr]
            eigenvectors=eigenvectors[:,indarr]
            # first axis = position, second axis = eigenvector, third axis = component
            eigenvectors=np.rollaxis(np.reshape(eigenvectors,(len(m._zp),6,self._neig)),2,1)
            # first axis = position, second axis = eigenvector, value = normsq
            normsq=(np.sum(abs(eigenvectors)**2,axis=2).T/self._mesh._dzm).T
            # first axis = component, second axis = eigenvector, value = compweight
            weight=np.sum((abs(eigenvectors)**2).T,axis=2)
            #psi_out[:,:]=(1/np.sqrt(self._mesh._dzm))*eigenvectors.T
            uens+=[energies]
            #psis+=[psi_out]
            normsqs+=[normsq]
            weights+=[weight]
        uens,normsqs,weights=np.array(uens), np.array(normsqs), np.array(weights)
        E_i=ConstantFunction(m,np.tile(np.atleast_3d(uens),len(m.zp)))
        eta=np.rollaxis((E_i-m.EF)/kT.tpf(),2,1)
        m['p']=np.sum(1/(2*np.pi)*np.trapz(kt*(normsqs/(1+np.exp(-eta))).T,x=kt),axis=0)
        m['pderiv']=np.sum(1/(2*np.pi*kT.tpf())*np.trapz(kt*(normsqs*(np.exp(-eta))/(1+np.exp(-eta))**2).T,x=kt),axis=0)
        log("not blending",level="TODO")

        return kt,uens,normsqs,weights


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





