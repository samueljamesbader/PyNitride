from poissolve.mesh.functions import MaterialFunction, ConstantFunction, PointFunction
from poissolve.constants import hbar, kT, m0, eV
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import numpy as np

class SchrodingerSolver():
    def __init__(self,mesh,carriers=['electron','hole']):
        self._mesh=m=mesh

        print("Assuming all materials have only one electron band")
        print("Not quantizing holes")

        pterms=self._pterms={c:{} for c in carriers}
        for carrier,v in pterms.items():
            for b in mesh._layers[0].material['ladder',carrier].keys():
                v[b]={}
                v[b]['mz']=mz=MaterialFunction(m,['ladder',carrier,b,'mzs'])
                v[b]['center']=(hbar**2/(mz*m._dz)).to_point_function(interp='unweighted')/mesh._dzp
                v[b]['off']=-(hbar**2/(2*mz*m._dz *np.sqrt(m._dzp[:-1]*m._dzp[1:])))
                v[b]['mxys']=MaterialFunction(m,['ladder',carrier,b,'mxys'],pos='point')
                v[b]['g']=MaterialFunction(m,['ladder',carrier,'Gamma','g'],pos='point')
                v[b]['DE']=MaterialFunction(m,['ladder',carrier,'Gamma','DE'],pos='point')


    def solve(self):
        m=self._mesh

        for carrier,v in self._pterms.items():
            conc=0
            for b, bp in v.items():
                abbrev="_"+carrier[0]+"_"+b
                center=bp['center']+m['Ec']+bp['DE']
                A=diags([bp['off'],center,bp['off']],[-1,0,1])
                w,v=eigsh(A,k=3,sigma=np.min(m['Ec']+bp['DE'])-1)
                psi=(1/np.sqrt(m._dzp))*v.T

                m['Psi_i'+abbrev]=Psi_i=PointFunction(m,psi)
                m['E_i'+abbrev]=E_i=ConstantFunction(m,w)

                meff=1/((Psi_i**2/bp['mxys']).integrate()[:,-1])
                #print("meff")
                #print(meff/m0)
                #### THIS ONLY MAKES SENSE FOR ELECTRONS
                conc+=(bp['g']/(2*np.pi)*kT/hbar**2)*\
                      np.sum(meff*(psi**2*(np.log(1+np.exp(-(E_i-m['EF'])/kT)))).T,axis=1)

                #print('hi')
            m[{'electron':'n','hole':'p'}[carrier]]=conc
        #mpl.figure()
        #mpl.plot(m.zp,m['n'].integrate()/(1e13/cm**2))
        #mpl.ylabel('Cum e-charge [1e13/cm^2]')
        print(w)


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
    m['mqV']=PointFunction(m,np.choose(1*(z>xc)+1*(z>xc+xb)+1*(z>xc+xb+xw),
                                       [F*(z-xc),
                                        0*z,
                                        F*(z-(xc+xb)),
                                        F*xw+0*z]))-.6*eV
    m['Ec']=m['mqV']+m['DEc']
    m['EF']=ConstantFunction(m,0)
    sm=m.submesh([5,30])
    #sm.plot_mesh()
    sm.plot_function('Ec')
    SchrodingerSolver(sm,carriers=['electron']).solve()
    from poissolve.visual import plot_wavefunctions
    mpl.figure()
    plot_wavefunctions(sm)
    mpl.show()
