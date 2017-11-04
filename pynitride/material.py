from pynitride.paramdb import pmdb, K, hbar, m_e, nm
from pynitride.mesh import MidFunction, PointFunction, Function, SubMesh
import numpy as np

class MaterialSystem():

    def __init__(self):
        self._attrs={
            'eps':      self.vergard('dielectric.eps'),
            'DE' :      self.vergard('DE'),
            'Eg':       self.bandedge_params,
            'E0-Ev':    self.bandedge_params,
            'Ec-E0':    self.bandedge_params,
        }
        self._defaults={
            'T':    300 *K
        }

    def surface_barrier(self,m):
        return self.vergard('surface={}.electronbarrier'.format(m._boundary[0]))(m,None)[0]

    def vergard(self,lookup):
        interpdict={k:pmdb["material="+v+"."+lookup] for k,v in self._vergardbasis.items()}
        def prop(mesh,key):
            molefracs={k:mesh[k] for k in interpdict.keys() if k is not None}
            val=(1-sum(v for v in molefracs.values()))*interpdict[None]
            for k,v in molefracs.items():
                val+=interpdict[k]*v
            if key is not None:
                mesh[key]=val
            return val
        return prop

    def polarization(self,mesh,key):
        raise NotImplementedError

    def append_dopants(self,dopants):
        for dn in dopants:
            types=[pmdb['material='+mat+'.dopant='+dn+'.type'] for mat in self._vergardbasis.values()]
            assert len(set(types))==1,\
                "Only one type (Donor/Acceptor) allowed for a dopant ("+dn+") in a material system."
        for prop in ['E','g']:
            self._attrs.update({dn+'.'+prop: self.vergard(dn+'.'+prop) for dn in dopants})
        self._dopants=[d+types[0] for d in dopants]
        self._defaults.update({d:0 for d in self._dopants})

    def get(self,mesh,item):
        if item in self._attrs:
            return self._attrs[item](mesh,item)
    def __contains__(self, item):
        return item in self._attrs

    def bulk(matsys,**kwargs):
        class BulkMaterial():
            def __init__(self,**kwargs):
                self._matsys=matsys
                self.mesh=self
                self._funcs={k:np.array([v]) for k,v in kwargs.items()}
                if 'exx' not in self._funcs:
                    self._funcs['exx']=0
                if 'eyy' not in self._funcs:
                    self._funcs['eyy']=0
            def __getitem__(self,item):
                if item in self._funcs:
                    return self._funcs[item]
                elif item in self._matsys._attrs:
                    return self._matsys.get(self,item)
            def __setitem__(self, key, value):
                self._funcs[key]=value
            def get(self,item):
                return self.__getitem__(item)
            def __getattr__(self, item):
                return self.__getitem__(item)
        return BulkMaterial(**kwargs)



class Wurtzite(MaterialSystem):
    def __init__(self):
        super().__init__()
        self._attrs.update({
            'exx':      self.strain,
            'eyy':      self.strain,

            'Psp':      self.vergard('polarization.Psp'),
            'e33':      self.vergard('polarization.e33'),
            'e31':      self.vergard('polarization.e31'),
            'C13':      self.vergard('stiffness.C13'),
            'C33':      self.vergard('stiffness.C33'),
            'P'  :      self.polarization,
            'DP' :      self.polarization,

            'medos':    self.smcls_band_params,
            'mexy':     self.smcls_band_params,
            'mez':      self.smcls_band_params,
            'cDE':      self.smcls_band_params,
            'eg':       self.smcls_band_params,
            'mhdos':    self.smcls_band_params,
            'mhxy':     self.smcls_band_params,
            'mhz':      self.smcls_band_params,
            'vDE':      self.smcls_band_params,
            'hg':       self.smcls_band_params,

            'A1':       self.kp_params,
            'A2':       self.kp_params,
            'A3':       self.kp_params,
            'A4':       self.kp_params,
            'A5':       self.kp_params,
            'A6':       self.kp_params,
            'D1':       self.kp_params,
            'D2':       self.kp_params,
            'D3':       self.kp_params,
            'D4':       self.kp_params,
            'D5':       self.kp_params,
            'D6':       self.kp_params,
            'DeltaSO':  self.kp_params,
            'DeltaCR':  self.kp_params,
            'Delta1':  self.kp_params,
            'Delta2':  self.kp_params,
            'Delta3':  self.kp_params,
            'a1':       self.kp_params,
            'a2':       self.kp_params,
        })

    def polarization(self,m,key):
        m['P']=m.ztrans*(m.Psp+m.e31*(m.exx+m.eyy)+m.e33*m.ezz)
        m['DP']=-m.P.differentiate(fill_value=0)
        return m[key]

    def bandedge_params(self,m,key):
        Eg0=self.vergard('conditions=relaxed.varshni.Eg0')(m,None)
        alpha=self.vergard('conditions=relaxed.varshni.alpha')(m,None)
        beta=self.vergard('conditions=relaxed.varshni.beta')(m,None)
        Eg_re=Eg0-alpha*m.T**2/(m.T+beta)

        s=(m.exx+m.eyy)/2
        Sigma2=(m.D1+m.D3)*m.ezz+(m.D2+m.D4)*2*s
        Sigmac=(m.a1+m.D1)*m.ezz+(m.a2+m.D2)*2*s

        m['Eg']=Eg_re + Sigmac-Sigma2
        m['E0-Ev']=Eg_re/2  -Sigma2
        m['Ec-E0']=Eg_re/2  +Sigmac

        # For kp this is the thing to add to Ev before solving Hamiltonian
        # Ev already includes (1) strain shift of VB but I want that in the Hamiltonian,
        # so subtract that out from Ev to get an Ev_raw.
        # Also, the bandedge in the Hamiltonian is not zero, it's max(Delta1+Delta2,Delta1-Delta2,0)
        # So let's set Ev_raw to Ev minus that amount to make sure the bulk solution top energy is Ev
        m['EvOffset']=-Sigma2-np.maximum(m.Delta1+m.Delta2,m.Delta1-m.Delta2,MidFunction(m,0))

        return m[key]

    def smcls_band_params(self,m,key):
        print("Using explicit masses from file")
        m['eg']=MidFunction(m,2)
        m['hg']=MidFunction(m,2)
        m['medos']=np.atleast_2d(
            self.vergard('carrier=electron.band=.mdos')(m,None))
        m['mhdos']=MidFunction(m,np.vstack([
            self.vergard('carrier=hole.band=HH.mdos')(m,None),
            self.vergard('carrier=hole.band=LH.mdos')(m,None),
            self.vergard('carrier=hole.band=CH.mdos')(m,None)]))
        m['mez']=np.atleast_2d(
            self.vergard('carrier=electron.band=.mzs')(m,None))
        m['mhz']=MidFunction(m,np.vstack([
            self.vergard('carrier=hole.band=HH.mzs')(m,None),
            self.vergard('carrier=hole.band=LH.mzs')(m,None),
            self.vergard('carrier=hole.band=CH.mzs')(m,None)]))
        m['mexy']=np.atleast_2d(
            self.vergard('carrier=electron.band=.mxys')(m,None))
        m['mhxy']=MidFunction(m,np.vstack([
            self.vergard('carrier=hole.band=HH.mxys')(m,None),
            self.vergard('carrier=hole.band=LH.mxys')(m,None),
            self.vergard('carrier=hole.band=CH.mxys')(m,None)]))
        m['cDE']=np.atleast_2d(
            self.vergard('carrier=electron.band=.DE')(m,None))
        m['vDE']=MidFunction(m,np.vstack([
            self.vergard('carrier=hole.band=HH.DE')(m,None),
            self.vergard('carrier=hole.band=LH.DE')(m,None),
            self.vergard('carrier=hole.band=CH.DE')(m,None)]))
        return m[key]

    def kp_params(self,m,key):
        self.vergard('kp.A1')(m,'A1')
        self.vergard('kp.A2')(m,'A2')
        self.vergard('kp.A3')(m,'A3')
        self.vergard('kp.A4')(m,'A4')
        self.vergard('kp.A5')(m,'A5')
        self.vergard('kp.A6')(m,'A6')
        self.vergard('kp.D1')(m,'D1')
        self.vergard('kp.D2')(m,'D2')
        self.vergard('kp.D3')(m,'D3')
        self.vergard('kp.D4')(m,'D4')
        self.vergard('kp.D5')(m,'D5')
        self.vergard('kp.D6')(m,'D6')
        self.vergard('kp.DeltaSO')(m,'DeltaSO')
        self.vergard('kp.DeltaCR')(m,'DeltaCR')
        self.vergard('kp.a1')(m,'a1')
        self.vergard('kp.a2')(m,'a2')
        m['Delta1']=m.DeltaCR
        m['Delta2']=m['Delta3']=1/3*m.DeltaSO
        return m[key]

    def kp_Cmats(self,m,kx,ky):
        U=MidFunction(m,hbar**2/(2*m_e))
        O=MidFunction(m,0)

        Atwid=m.A2+m.A4-U; Ahat=m.A1+m.A3-U
        L1=m.A5+Atwid; L2=m.A1-U
        M1=-m.A5+Atwid; M2=Ahat; M3=m.A2-U
        N1p=3*m.A5-Atwid; N1m=-m.A5+Atwid
        N2p=np.sqrt(2)*m.A6-Ahat
        N2m=Ahat
        l1=m.D2+m.D4+m.D5; l2=m.D1
        m1=m.D2+m.D4-m.D5; m2=m.D1+m.D3; m3=m.D2
        n1=-2*m.D5; n2=np.sqrt(2)*m.D6
        L1u=L1+U; L2u=L2+U
        M1u=M1+U; M2u=M2+U; M3u=M3+U
        Delta1=m.Delta1;Delta2=m.Delta2;Delta3=m.Delta3

        def double(arr):
            n=arr.shape[0]
            out=Function(m,value=np.zeros((2*n,2*n,arr.shape[2]),dtype='complex'),dtype='complex',pos=arr.pos)
            out[:n,:n,:]=arr
            out[n:,n:,:]=arr
            return out

        Cmats=[]
        for kx,ky in zip(kx,ky):
            C0= \
                (double(MidFunction(m,[
                    [kx*L1u*kx+ky*M1u*ky , kx*N1p*ky+ky*N1m*kx,         O           ],
                    [ky*N1p*kx+kx*N1m*ky , kx*M1u*kx+ky*L1u*ky,         O           ],
                    [         O          ,          O         ,  kx*M3u*kx+ky*M3u*ky]]))+ \
                 MidFunction(m,[
                     [   Delta1,   -1j*Delta2,          O,          O,         O,     Delta3],
                     [1j*Delta2,       Delta1,          O,          O,         O, -1j*Delta3],
                     [        O,            O,          O,    -Delta3, 1j*Delta3,          O],
                     [        O,            O,    -Delta3,     Delta1, 1j*Delta2,          O],
                     [        O,            O, -1j*Delta3, -1j*Delta2,    Delta1,          O],
                     [   Delta3,    1j*Delta3,          O,          O,         O,          O]],dtype='complex')+ \
                 double(MidFunction(m,[
                     [l1*m.exx+m1*m.eyy+m2*m.ezz,       n1*m.exy,                   n2*m.exz],
                     [       n1*m.exy,        m1*m.exx+l1*m.eyy+m2*m.ezz,           n2*m.eyz],
                     [       n2*m.exz,              n2*m.eyz,            m3*m.exx+m3*m.eyy+l2*m.ezz]]))).tpf()

            Cl= m.ztrans*\
                double(MidFunction(m,[
                    [           O     ,             O    ,       kx*N2p     ],
                    [           O     ,             O    ,       ky*N2p     ],
                    [       kx*N2m    ,         ky*N2m   ,         O        ]])).tpf()
            Cr= m.ztrans*\
                double(MidFunction(m,[
                    [           O     ,             O    ,       N2m*kx     ],
                    [           O     ,             O    ,       N2m*ky     ],
                    [       N2p*kx    ,         N2p*ky   ,         O        ]])).tpf()

            C2= \
                double(MidFunction(m,[
                    [           M2u   ,             O    ,        O     ],
                    [           O     ,           M2u    ,        O     ],
                    [           O     ,              O   ,       L2u    ]]))
            Cmats+=[[C0,Cl,Cr,C2]]
        return Cmats


    def strain(self,m,key):
        a=self.vergard('conditions=relaxed.lattice.a')(m.subs.mesh,None)[0]
        a0=self.vergard('conditions=relaxed.lattice.a')(m,None)
        m['exx']=m['eyy']=(a-a0)/a0
        m['ezz']=-2*m.C13/m.C33*m['exx']
        m['exy']=m['eyz']=m['exz']=MidFunction(m,0)
        return m[key]


#class AlGaInN(Wurtzite):
#    def __init__(self):
#
#        self._vergardbasis={
#            'x': 'AlN',
#            'y': 'InN',
#            None: 'GaN',
#        }
#
#        super().__init__()
#
#        self._defaults.update({
#            'x': 0
#            'y': 0
#        })

class AlGaN(Wurtzite):
    def __init__(self):
        self._vergardbasis={
            'x': 'AlN',
            None: 'GaN',
        }

        super().__init__()

        self._defaults.update({
            'x': 0
        })
        self.append_dopants(['Si','Mg'])

class SamGaN(Wurtzite):
    def __init__(self):
        self._vergardbasis={
            'x': 'Sam',
            None: 'GaN',
        }

        super().__init__()

        self._defaults.update({
            'x': 0
        })
        self.append_dopants(['Si','Mg'])

if __name__=="__main__":
    GaN=AlGaN.bulk(x=0,y=0)
    AlN=AlGaN.bulk(x=1,y=0)
    Al5Ga5N=AlGaN.bulk(x=.5,y=0)
    print("GaN eps: ",GaN.get("eps"))
    print("AlN eps: ",AlN.get("eps"))
    print("Al5Ga5N eps: ",Al5Ga5N.get("eps"))
    print("GaN pol: ",GaN.get("P"))

