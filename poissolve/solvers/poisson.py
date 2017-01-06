import numpy as np
from poissolve.constants import q
from poissolve.mesh_functions import MidFunction, MaterialFunction, PointFunction
from poissolve.maths.tdma import tdma

class PoissonSolver():
    def __init__(self, mesh):
        self._mesh = mesh


        # ARE THESE NECESSARY
        mesh['D']= MidFunction(mesh)
        mesh['E']= MidFunction(mesh)
        eps=mesh['eps']= MaterialFunction(mesh, 'eps')
        mesh['mqV']= PointFunction(mesh, 0.0)
        mesh['DEc']= MaterialFunction(mesh, "DEc", pos='point')
        mesh['Ec']= PointFunction(mesh)
        mesh['Eg']= MaterialFunction(mesh, "Eg", pos='point')
        mesh['Ev']= PointFunction(mesh)

        mesh['arho2']= PointFunction(mesh)


        self._left=np.empty(len(mesh._z))
        self._right=np.empty(len(mesh._z))
        self._left[1:]=eps/(mesh._dz * mesh._dzp[1:])
        self._right[:-1]=eps/(mesh._dz * mesh._dzp[:-1])
        self._center=-MidFunction(mesh,eps/mesh._dz).to_point_function(interp='unweighted')/mesh._dzp
        #self._center=np.zeros(len(mesh.z))
        #self._center[1:-1]=-1/(mesh._dz[1:]*mesh._dz[:-1])
        self._center[-1]=self._center[-2]

        self._left[:2]=0
        self._right[0]=0
        self._right[-1:]=0
        self._center[0]=1
        self._center[1:-1]*=2

        self._mqV_temp = PointFunction(mesh)  # temp

    def solve(self,damp=0):
        m=self._mesh
        qrho=q*m['rho']
        qrho[0]=0
        #print(self._left)
        #print(self._right)
        #print(self._center)
        #print(rho)
        temp=tdma(self._left,self._center,self._right,qrho)
        m['mqV']=(damp)*m['mqV']+(1-damp)*temp
        self._update_others()
        mqV=m['mqV']
        m['E']=mqV.differentiate()
        m['D']=MidFunction(m,MaterialFunction(m,'eps')*m['E'])
        m['arho2']=m['D'].differentiate()

    def _update_others(self):
        m=self._mesh
        phib=3
        m['Ec']=m['mqV']+m['EF'][0]+phib+m['DEc']
        m['Ev']=m['Ec']-m['Eg']

    def isolve(self,visual=False):
        m=self._mesh
        qrho=q*m['rho']
        qrho[0]=0



        diag = -self._center + q * self._mesh['rhoderiv']
        diag[0] -= q * m['rhoderiv'][0]
        diag[-1] -= q * m['rhoderiv'][-1]

        a=-self._left
        b=diag
        c=-self._right
        #d= (q*self._rhoprev - qrho) #bad
        d= (q*m['arho2'] - qrho)
        d[0]=0
        d[-1]=-m['D'][-1]
        #assert np.isclose(-D[-1],nonuniformmesh['rho'][-1])

        import numpy as np
        if visual:
            import matplotlib.pyplot as mpl
            mpl.figure()
            mpl.subplot(311)
            mpl.plot(m.z,qrho-q*self._rhoprev)
            print(np.max(np.abs(qrho-q*self._rhoprev)))
            mpl.title('rho- rhoprev')
            mpl.subplot(312)
            mpl.plot(m.z,m['rhoderiv'])
            mpl.title('rhoderiv')
            mpl.tight_layout()

        dqmV=tdma(a,b,c,d)
        #print(dqmV[-15:])
        m['mqV']+=dqmV
        self._update_others()
        self._rhoprev=m['rho'].copy()


        mqV=m['mqV']
        m['E']=mqV.differentiate()
        m['D']=MidFunction(m,MaterialFunction(m,'eps')*m['E'])
        m['arho2']=m['D'].differentiate()


        return np.max(np.abs(dqmV))#np.sum(np.abs(dqmV))/np.sum(np.abs(m['mqV']))


