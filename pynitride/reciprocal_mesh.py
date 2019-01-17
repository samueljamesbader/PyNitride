import numpy as np



class KMesh2D:
    def __init__(self,kx,ky):
        self.kx1=kx
        self.kx1p=kx[kx>=0]
        self.ky1=ky
        self.ky1p=ky[ky>=0]
        self.lkx=len(self.kx1)
        self.lky=len(self.ky1)
        self.KX,self.KY=np.meshgrid(kx,ky)
        self.KT=np.stack([self.KX,self.KY],axis=2)
        self.kt=self.conv2flat(self.KT)
        self.kx=self.kt[:,0]
        self.ky=self.kt[:,1]
        if len(kx)>1:
            self.dkx=kx[1]-kx[0]
        if len(ky)>1:
            self.dky=ky[1]-ky[0]

    def conv2grid(self,arr):
        return np.reshape(arr,[self.lky,self.lkx]+list(arr.shape[1:]))
    def conv2flat(self,arr):
        return np.reshape(arr,[np.prod(arr.shape[:2])]+list(arr.shape[2:]))
    def intflat(self,arr):
        ig= self.conv2grid(arr).T
        return np.trapz(np.trapz(ig,dx=self.dky),dx=self.dkx).T
    def along(self,arr,dir='x',input='guess',onesided=True):
        assert input=='flat' or (input=='guess' and arr.shape[0]==self.lkx*self.lky)
        if dir=='x':
            iy=np.argmax(self.ky1==0)
            assert (self.ky1[iy]==0), "0 not in ky"
            ix = np.argmax(self.kx1>=0) if onesided else 0
            return arr[(iy*self.lkx+ix):(iy*self.lkx+self.lkx)]
        elif dir=='y':
            ix=np.argmax(self.kx1==0)
            assert (self.kx1[ix]==0), "0 not in kx"
            iy=np.argmax(self.ky1>=0) if onesided else 0
            return arr[(ix+iy*self.lkx)::self.lkx]
