from poissolve.constants import q
from poissolve.mesh_functions import MidFunction, MaterialFunction, PointFunction
from poissolve.maths.tdma import tdma

class PoissonSolver():
    def __init__(self, mesh):
        self._mesh = mesh
        mesh.add_function('D', MidFunction(mesh))
        mesh.add_function('E', MidFunction(mesh))
        eps=mesh.add_function('eps', MaterialFunction(mesh, 'eps')).array
        mesh.add_function('mqV', PointFunction(mesh, arr=0.0))
        mesh.add_function('DEc', MaterialFunction(mesh, "DEc", pos='point'))
        mesh.add_function('Ec', PointFunction(mesh))
        mesh.add_function('Eg', MaterialFunction(mesh, "Eg", pos='point'))
        mesh.add_function('Ev', PointFunction(mesh))

        mesh.add_function('arho2', PointFunction(mesh))

        import numpy as np

        self._left=np.empty(len(mesh._z))
        self._right=np.empty(len(mesh._z))
        self._left[1:]=eps/(mesh._dz * mesh._dzp[1:])
        self._right[:-1]=eps/(mesh._dz * mesh._dzp[:-1])
        self._center=-MidFunction(mesh=mesh,arr=eps/mesh._dz).to_point_function(interp='unweighted').array/mesh._dzp
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
        qrho=q*m['rho'].array
        qrho[0]=0
        #print(self._left)
        #print(self._right)
        #print(self._center)
        #print(rho)
        temp=tdma(self._left,self._center,self._right,qrho)
        m['mqV']=(damp)*m['mqV'].array+(1-damp)*temp
        self._update_others()
        mqV=m['mqV']
        m['E']=mqV.differentiate().array
        m['D']=MidFunction(m,MaterialFunction(m,'eps').array*m['E'].array).array
        m['arho2'].array=m['D'].differentiate().array

    def _update_others(self):
        m=self._mesh
        phib=3
        m['Ec']=m['mqV'].array+m['EF'][0]+phib+m['DEc'].array
        m['Ev']=m['Ec'].array-m['Eg'].array

    def isolve(self,visual=False):
        m=self._mesh
        qrho=q*m['rho'].array
        qrho[0]=0



        diag = -self._center + q * self._mesh['rhoderiv']
        diag[0] -= q * m['rhoderiv'][0]
        diag[-1] -= q * m['rhoderiv'][-1]

        a=-self._left
        b=diag
        c=-self._right
        #d= (q*self._rhoprev - qrho) #bad
        d= (q*m['arho2'].array - qrho)
        d[0]=0
        d[-1]=-m['D'][-1]
        #assert np.isclose(-D[-1],nonuniformmesh['rho'].array[-1])

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
        m['mqV'].array+=dqmV
        self._update_others()
        self._rhoprev=m['rho'].array.copy()


        mqV=m['mqV']
        m['E']=mqV.differentiate().array
        m['D']=MidFunction(m,MaterialFunction(m,'eps').array*m['E'].array).array
        m['arho2'].array=m['D'].differentiate().array


        return np.max(np.abs(dqmV))#np.sum(np.abs(dqmV))/np.sum(np.abs(m['mqV'].array))


    # def solve(self, damp=0):
    #     self._rho.integrate(flipped=True, output=self._D)
    #     self._E.array = self._D.array / self._mesh['eps']
    #     self._E.integrate(output=self._mqV_temp)  # temp
    #     # self._mqV_temp.array-=self._mqV_temp.array[0]
    #     self._mqV.array = damp * self._mqV.array + (1 - damp) * self._mqV_temp.array
    #
    #     self._mesh['Ec'] = self._mqV.array + self._mesh['EF'][0] + phib + self._DEc.array
    #     self._mesh['Ev'] = self._Ec.array - self._Eg.array
    #
    #     self._rhoprev = self._rho.array.copy()

    def slowsolve(self):
        diag = 2 * self._eps_o_d2_point.array

        lhs = sp.diags([self._km1, diag, self._k1], offsets=[-1, 0, 1])
        rhs = q * (-self._rho.array)
        rhs[0] = 0

        # rhs*=10000
        with sam.printoptions(precision=5, suppress=True):
            print(lhs.todense())
            print(rhs)
        # mpl.imshow(lhs.todense())
        # return


        self._mqV.array, info = cg(lhs, rhs)  # ,tol=1e-6)
        print("info: " + str(info))
        self._mesh['E'] = self._mesh.get_function('mqV').differentiate().array

        self._rhoprev = self._rho.array.copy()

        return lhs, rhs

    # def isolve(self):
    #     diag = 2 * self._eps_o_d2_point.array + q * self._mesh['rhoderiv']
    #     diag[0] -= q * self._mesh['rhoderiv'][0]
    #
    #     lhs = sp.diags([self._km1, diag, self._k1], offsets=[-1, 0, 1])
    #     rhs = q * (self._rhoprev - self._rho.array)
    #     rhs[0] = 0
    #
    #     # mpl.imshow(lhs.todense())
    #     # return
    #     dmqV, info = cg(lhs, rhs, tol=1e-6)
    #     print("info: " + str(info))
    #     self._mqV.array += dmqV
    #     self._mesh['E'] = self._mesh.get_function('mqV').differentiate().array
    #
    #     self._rhoprev = self._rho.array.copy()