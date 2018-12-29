import numbers

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from pynitride.paramdb import k,hbar,q,m_e, cm, pi
from pynitride.maths import tdma, fd12, fd12p, idd,iddd
from pynitride.mesh import MaterialFunction, PointFunction, ConstantFunction, MidFunction, SubMesh
from collections import OrderedDict
from operator import mul
from functools import reduce, lru_cache

from pynitride.visual import log, sublog





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
        m=self._mesh = mesh

        alldopants=sum((mb.matsys._dopants for mb in m._matblocks),[])
        self._donors    =[d for d in alldopants if d.endswith("Donor")]
        self._acceptors =[d for d in alldopants if d.endswith("Acceptor")]

        for d in self._donors+self._acceptors:
            if np.any(np.diff(m[d+"g"])!=0):
                print(d,m[d+"g"])
                raise Exception("Non-uniform g not working yet because idd takes one g")

        surface=m._boundary[0]
        if isinstance(surface,numbers.Real):
            self._phib=surface
        else:
            self._phib=m._matblocks[0].matsys.surface_barrier(m._matblocks[0].mesh)


        m.ensure_function_exists('rho',0)
        m.ensure_function_exists('rhoderiv',0)
        m.ensure_function_exists('fixedcharge',0)

        m.ensure_function_exists('Ndp',0)
        m.ensure_function_exists('Nam',0)
        m.ensure_function_exists('Ndpderiv',0)
        m.ensure_function_exists('Namderiv',0)

        m.ensure_function_exists('DP',0)
        m.ensure_function_exists('phi',0)

        if len(m.zm)>1:
            self._left=np.empty(len(m.zp))
            self._right=np.empty(len(m.zp))
        self.update_epsfactor(epsfactor=1)
        self._update_others(-m['phi'])
        self.ionized_dopants(gotzloop=False)

    def update_epsfactor(self,epsfactor):
        self._eps=epsfactor*self._mesh.eps
        if len(self._mesh.zm)>1:
            self.assemble_fixed()

    def assemble_fixed(self):
        m=self._mesh
        eps=self._eps
        self._left[1:]=eps/(m.dzp * m.dzm[1:])
        self._right[:-1]=eps/(m.dzp * m._dzm[:-1])
        self._center= -(eps/m.dzp).tpf(interp='unweighted') / m.dzm
        self._center[-1]=self._center[-2]
        self._left[-1]/=2
        self._left[:2]=0
        self._right[0]=0
        self._right[-1:]=0
        self._center[0]=1
        self._center[1:-1]*=2


    def ionized_dopants(self,gotzloop=False):
        # Tiwari Compound Semiconductor Devices pg31-32
        m=self._mesh
        kT=k*m.T

        m['Ndp']=0
        m['Ndpderiv']=0
        for d in self._donors:
            g=m[d+"g"][0]
            conc=m[d+"Conc"]
            eta=((m.EF-m.Ec).tmf()+m[d+"E"])/kT
            m['Ndp']+=(conc*idd(eta,g)).tpf()
            m['Ndpderiv']+=(conc/kT*iddd(eta,g)).tpf()

        if gotzloop is False:
        #if 1:
            #m['Nam']=0
            Nam=PointFunction(m,value=0)
            m['Namderiv']=0
            for d in self._acceptors:
                g=m[d+"g"][0]
                conc=m[d+"Conc"]
                eta=((m.Ev-m.EF).tmf()+m[d+"E"])/kT
                #m['Nam']+=(conc*idd(eta,g)).tpf()
                Nam+=(conc*idd(eta,g)).tpf()
                m['Namderiv']-=(conc/kT*iddd(eta,g)).tpf()

            m['Nam']=Nam
        else:
            while True:
                Nam=PointFunction(m,value=0)
                m['Namderiv']=0
                for d in self._acceptors:
                    g=m[d+"g"][0]
                    conc=m[d+"Conc"]
                    f=2.1828
                    gotz=m[d+'gotzshift']=-m.gotz*f*(q**2)/(4*pi*m.eps)*(m.Nam.tmf())**(1/3)
                    #gotz=0
                    eta=((m.Ev-m.EF).tmf()+m[d+"E"]+gotz)/kT
                    Nam+=(conc*idd(eta,g)).tpf()
                    m['Namderiv']-=(conc/kT*iddd(eta,g)).tpf()
                    #log("Gotz max {:.6f}".format(float(np.max(np.abs(gotz)))),'debug')
                    #log("Namdiffmax {:.2e}".format(float(np.max(np.abs((Nam-m.Nam))))),'debug')
                if np.max(np.abs((Nam-m.Nam)))<1e16/cm**3:
                    m['Nam']=Nam
                    break
                m['Nam']=Nam



    def solve(self):
        m=self._mesh
        self.ionized_dopants()
        m['DP']=-m.P.differentiate(fill_value=0)
        if 'p' in m:
            p=m.p
            pderiv=m.pderiv
        else:
            p=pderiv=0
        if 'n' in m:
            n=m.n
            nderiv=m.nderiv
        else:
            n=nderiv=0
        m['rho']=p-n+m.Ndp-m.Nam+m.DP+m.fixedcharge
        m['rhoderiv']=pderiv-nderiv+m.Ndpderiv-m.Namderiv


        qrho=q*m.rho
        qrho[0]=0
        print("lcr")
        print(sum(self._left))
        print(sum(self._center))
        print(sum(self._right))
        mqV=PointFunction(m,tdma(self._left,self._center,self._right,qrho))
        err=np.max(np.abs(m.phi+mqV))
        m['phi']=-mqV
        self._update_others(mqV)
        return err

    def _update_others(self, mqV):
        m=self._mesh
        self._mqV=mqV
        m['phi']=-mqV
        m['Ec']=mqV+m.EF[0]+self._phib+m.DE.tpf()-m.DE[0]+m['Ec-E0'].tpf()-m['Ec-E0'][0]
        m['Ev']=m.Ec-m.Eg.tpf()
        m['E']=mqV.differentiate()
        m['D']=self._eps*m['E']
        self._arho2=m['D'].differentiate()

    def store_state(self):
        self._storedmqV=self._mqV.copy()
    def restore_state(self):
        self._mqV=self._storedmqV
        self._update_others(self._mqV)

    def isolve(self,visual=False,activation=1):
        log("Poisson iSolve",level="debug")
        m=self._mesh
        m['DP']=-m.P.differentiate(fill_value=0)
        self.ionized_dopants()
        if 'p' in m:
            p=m.p
            pderiv=m.pderiv
        else:
            p=pderiv=0
        if 'n' in m:
            n=m.n
            nderiv=m.nderiv
        else:
            n=nderiv=0
        m['rho']=p-n+m.Ndp-m.Nam+m.DP+m.fixedcharge
        m['rhoderiv']=pderiv-nderiv+m.Ndpderiv-m.Namderiv
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

        d= (q*self._arho2 - qrho)
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
        self._dqmV=activation*dqmV
        #print(dqmV[-15:])
        self._mqV+=self._dqmV
        self._update_others(self._mqV)
        self._rhoprev=m['rho'].copy() # delete this

        return np.max(np.abs(dqmV))#np.sum(np.abs(dqmV))/np.sum(np.abs(m['mqV']))

    def shorten_last_step(self,factor):
        self._update_others(self._mqV-(1-factor)*self._dqmV)

class Equilibrium():

    def __init__(self,mesh):
        self._mesh=mesh
        mesh.ensure_function_exists('EF',0)

    def solve(self):
        self._mesh['EF']=0

#class ChargeNeutral():
#
#    def __init__(self,mesh,carriersolvers=[],resolve_carriers=False):
#        self._mesh=mesh
#        self._mesh.ensure_function_exists('EF',value=0)
#        #self._mesh.ensure_function_exists('phi',0)
#        self._cs=carriersolvers
#        self._ps=PoissonSolver(mesh)
#        if resolve_carriers:
#            for cs in self._cs: cs.solve()
#
#    def solve(self, check='integrated', tol=None):
#        with sublog("Neutralizing charge","debug"):
#            m=self._mesh
#            if tol is None:
#                tol={'integrated':1e6/cm**2,'mean':1e9/cm**3}[check]
#            if check=='mean':
#                tol*=m.thickness
#            kT=k*np.max(m.T)
#            while True:
#                for cs in self._cs:
#                    cs.repopulate()
#                self._ps.ionized_dopants()
#                rho=(m.p-m.n+m.Ndp-m.Nam+m.DP).integrate(definite=True)
#                if abs(rho)<tol: break
#                rhoderiv=(m.pderiv-m.nderiv+m.Ndpderiv-m.Namderiv).integrate(definite=True)
#                log("Rho: {:.2e}        Rho' {:.2e}".format(float(rho),float(rhoderiv)),"debug")
#                dEF=np.sign(rho)*min(np.abs(rho/rhoderiv),kT)
#                m['EF']+=dEF


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


class SelfConsistentLoop():
    def __init__(self,fieldsolvers=[],carriersolvers=[]):
        self._fs=fieldsolvers
        self._cs=carriersolvers

    def isolve_fields(self, activation=1):
        return sum(fs.isolve(activation=activation) for fs in self._fs)
    def solve_fields(self):
        return sum(fs.solve() for fs in self._fs)
    def solve_carriers(self):
        [cs.solve_and_repopulate() for cs in self._cs]

    def loop(self, tol=1e-10, max_iter=100, min_activation=.1):
        adec=2
        with sublog("Starting SC loop"):
            err=np.inf
            i=0
            a=1
            while err>tol:
                if i>=max_iter:
                    raise Exception("Maximum iteration reached in SC loop")
                self.solve_carriers()
                errprev=err
                err=self.isolve_fields(activation=a)
                log("iter: {:3d}  err: {:.2e}  activ: {:g}".format(i,err,a))
                while err>errprev:
                    a/=adec
                    log("Retrying with Poisson activation={:g}".format(a))
                    if a<min_activation:
                        raise Exception("Couldn't reduce error in SC loop")
                    for fs in self._fs:fs.shorten_last_step(1/adec)
                    self.solve_carriers()
                    err=self.isolve_fields(activation=a)
                    log("       iter: {:3d}  err: {:.2e}".format(i,err))
                a=min(1.2*a,1)
                i+=1
            #log("Post-solve")
            #self.solve_fields()
            log("Loop finished in {:2d} iterations with err={:g}".format(i,err))

    def ramp_epsfactor(self, start=1e4, stop=1, dlefstart=.1, dlefmax=.5, dlefmin=.05, max_iter=10, tol=1e-6, min_activation=.1):
        with sublog("Starting eps factor ramp from {:g} to {:g}".format(start,stop)):
            lefstart=np.log10(start)
            lefstop=np.log10(stop)
            lef=lefstart

            dlefstart*=np.sign(lefstop-lefstart)
            dlef=dlefstart
            prevlef=None
            while True:
                ef=10**lef
                log("Eps factor: {:.2e}".format(ef))

                for fs in self._fs:
                    fs.store_state()
                    fs.update_epsfactor(ef)
                    self.isolve_fields()
                try:
                    self.loop(max_iter=max_iter, tol=tol, min_activation=min_activation)
                    if (lef-lefstop)<1e-9:
                        break

                    # Next lef
                    prevlef=lef
                    dlef=np.sign(dlef)*min(np.abs(2*dlef),np.abs(dlefmax))
                    lef=lef+dlef
                    # if we passed the end, go back
                    if np.sign(lef-lefstop)!=np.sign(lefstart-lefstop): lef=lefstop
                except Exception as e:
                    log("Failure: {}".format(str(e)))
                    if prevlef is None:
                        raise Exception("Failed at initial epsfactor")
                    ef=10**prevlef
                    log("Restoring at {:.2e}".format(ef))
                    for fs in self._fs:
                        fs.restore_state()
                        fs.update_epsfactor(ef)
                    self.solve_carriers()

                    dlef=dlef/2
                    if np.abs(dlef)<np.abs(dlefmin):
                        raise Exception("Eps factor step size too small")
                    lef=prevlef+dlef
                    # if we passed the end, go back
                    if np.sign(lef-lefstop)!=np.sign(lefstart-lefstop): lef=lefstop
            log("Done eps factor ramp")
