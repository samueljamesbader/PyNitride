from pynitride import pmdb, hbar, m_e, nm
from pynitride import MidFunction, NodFunction, Function, SubMesh
from pynitride import log
from pynitride.core.maths import double_mat
import numpy as np

class MaterialSystem():

    def __init__(self):
        """ Represents a parameterized material.

        """
        self._attrs={
            'eps':      self.vergard('dielectric.eps_z'),
            'epsperp':  self.vergard('dielectric.eps_perp'),
            'DE' :      self.vergard('DE'),
            'Eg':       self._bandedge_params,
            'E0-Ev':    self._bandedge_params,
            'Ec-E0':    self._bandedge_params,
        }
        self._defaults={}
        self._updates={}

    def surface_barrier(self,m):
        """ Returns the surface barrier height based on the top boundary of the mesh.

        Args:
            m: the mesh, should be the global mesh so there is a top boundary
        """
        return self.vergard('surface={}.electronbarrier'.format(m._boundary[0]))(m,None)[0]

    def vergard(self,lookup):
        """ Interpolates a parameter for the material system.

        The material using this function must have a vergard basis defined, eg

        .. code-block:: python

            self.vergard={ 'x': 'AlN', 'y':'InN', None: 'GaN'}

        where `'x', 'y'` are arbitrarily named mole-fraction variables that are defined on the mesh and the dict values
        are strings indicating the basis material for each mole-fraction variable.  The `None` key indicates the basis
        material for the `x=y=0` limit.  Each of the above materials must have the relevant lookup defined in the
        parameter database.

        Args:
            lookup: the string with which to query the parameter database for each material basis
        """
        interpdict={k:pmdb["material="+v+"."+lookup] for k,v in self.vergardbasis.items()}
        def prop(mesh,key):
            if len(interpdict.keys())==1:
                val=MidFunction(mesh,value=interpdict[None])
            else:
                molefracs={k:mesh[k] for k in interpdict.keys() if k is not None}
                val=(1-sum(v for v in molefracs.values()))*interpdict[None]
                for k,v in molefracs.items():
                    val+=interpdict[k]*v
            if key is not None:
                mesh[key]=val
            return val
        return prop

    def polarization(self,mesh,key):
        """ Populates the mesh with the polarization `P` and potentially other related functions

        Args:
            mesh: the mesh
            key: the key being sought

        """
        raise NotImplementedError

    def append_dopants(self,dopants):
        if not hasattr(self,'_dopants'): self._dopants=[]
        for dn in dopants:
            types=[pmdb['material='+mat+'.dopant='+dn+'.type'] for mat in self.vergardbasis.values()]
            assert len(set(types))==1,\
                "Only one type (Donor/Acceptor) allowed for a dopant ("+dn+") in a material system."
            self._dopants+=[dn+types[0]]
        for prop in ['E','g']:
            self._attrs.update({d+prop: self.vergard(dn+'.'+prop) for d,dn in zip(self._dopants,dopants)})
        self._defaults.update({d+"Conc":0 for d in self._dopants})

    def get(self,mesh,item):
        return self._attrs[item](mesh,item)
    def __contains__(self, item):
        return item in self._attrs

    def bulk(matsys,**kwargs):
        class BulkMaterial(matsys.__class__):
            def __init__(self,**kwargs):

                # Set up properties to impersonate both a MaterialSystem and a trivial Mesh
                self._matsys=self.matsys=matsys
                self.mesh=self
                self.zm=0
                self.ztrans=1
                self._matblocks=[self]
                self.ones_mid=1
                self.zeros_mid=0

                # Initialize as a MaterialSystem
                self._dopants=[]
                super().__init__()
                self._funcs=matsys._defaults.copy()

                # Default strains to 0, T to 300
                assert ('exx' not in kwargs) and ('eyy' not in kwargs)
                kwargs2={k:0 for k in ['exx','eyy','exy','exz','eyz','ezz']}
                kwargs2['T']=300
                kwargs2.update(kwargs)

                # Other specials supplied
                self._funcs.update({k:np.array(v) for k,v in kwargs2.items()})
                #try:
                #    self._funcs['ezz']=-2*self.C13/self.C33*self['exx']
                #except: pass

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
    def __init__(self,spin_splitting=0):
        """ Superclass for Wurtzite materials.

        Args:
            spin_splitting: an artificial splitting energy between spin-up and spin-down.  If a non-zero splitting
                energy is provided, the small terms in the k.p Hamiltonian which couple spin-up and spin-down are zeroed
                so that the spin_splitting will be exactly as specified.  This is convenient for scattering problems
                whenever spin is not important, because thereafter, one can simply ignore one half of the bands.
        """
        super().__init__()
        self._spin_splitting=spin_splitting

        self._updates.update({
            'strain': [self._bandedge_params,self.polarization],
            'temperature': [self._bandedge_params],
        })
        self._attrs.update({
            'Psp':      self.vergard('polarization.Psp'),
            'e33':      self.vergard('polarization.e33'),
            'e31':      self.vergard('polarization.e31'),
            'e51':      self.vergard('polarization.e51'),
            'C11':      self.vergard('stiffness.C11'),
            'C12':      self.vergard('stiffness.C12'),
            'C13':      self.vergard('stiffness.C13'),
            'C33':      self.vergard('stiffness.C33'),
            'C44':      self.vergard('stiffness.C44'),
            'P'  :      self.polarization,

            'medos':    self._smcls_band_params,
            'mexy':     self._smcls_band_params,
            'mez':      self._smcls_band_params,
            'cDE':      self._smcls_band_params,
            'eg':       self._smcls_band_params,
            'mhdos':    self._smcls_band_params,
            'mhxy':     self._smcls_band_params,
            'mhz':      self._smcls_band_params,
            'vDE':      self._smcls_band_params,
            'hg':       self._smcls_band_params,

            'EvOffset': self._bandedge_params,
            'EcOffset': self._bandedge_params,
            'A1':       self._kp_params,
            'A2':       self._kp_params,
            'A3':       self._kp_params,
            'A4':       self._kp_params,
            'A5':       self._kp_params,
            'A6':       self._kp_params,
            'D1':       self._kp_params,
            'D2':       self._kp_params,
            'D3':       self._kp_params,
            'D4':       self._kp_params,
            'D5':       self._kp_params,
            'D6':       self._kp_params,
            'DeltaSO':  self._kp_params,
            'DeltaCR':  self._kp_params,
            'Delta1':   self._kp_params,
            'Delta2':   self._kp_params,
            'Delta3':   self._kp_params,
            'a1':       self._kp_params,
            'a2':       self._kp_params,

            # Note: ignores bowing  of optical frequencies,
            # eg reported by https://doi.org/10.1063/1.121095
            'wLO_para': self.vergard('raman.wLO_para'),
            'wLO_perp': self.vergard('raman.wLO_perp'),
            'wTO_para': self.vergard('raman.wTO_para'),
            'wTO_perp': self.vergard('raman.wTO_perp'),
            'eps_inf':  self.vergard('dielectric.eps_inf'),

            'density':  self.vergard('density'),
        })

    def polarization(self,m,key):
        m['P']=m.ztrans*(m.Psp+m.e31*(m.exx+m.eyy)+m.e33*m.ezz)
        return m[key]

    def _bandedge_params(self,m,key=None):
        Eg0=self.vergard('conditions=relaxed.varshni.Eg0')(m,None)
        alpha=self.vergard('conditions=relaxed.varshni.alpha')(m,None)
        beta=self.vergard('conditions=relaxed.varshni.beta')(m,None)
        Eg_re=Eg0-alpha*m.T**2/(m.T+beta)

        # 3x3 valence band without strain
        H1=m.Delta1+m.Delta2
        H2=m.Delta1-m.Delta2
        H3=0
        E1=H1
        E2=((H2+H3)+np.sqrt((H2+H3)**2-4*(H2*H3-2*m.Delta3**2)))/2
        E3=((H2+H3)-np.sqrt((H2+H3)**2-4*(H2*H3-2*m.Delta3**2)))/2

        # This is the top of the kp valence band without strain
        EV0=MidFunction(m,np.max([m.Delta1+m.Delta2,m.Delta1-m.Delta2,m.zeros_mid],axis=0))

        # 3x3 valence band including strain
        H1=m.Delta1+m.Delta2+(m.D1+m.D3)*m.ezz+(m.D2+m.D4)*(m.exx+m.eyy)
        H2=m.Delta1-m.Delta2+(m.D1+m.D3)*m.ezz+(m.D2+m.D4)*(m.exx+m.eyy)
        H3=                  (m.D1     )*m.ezz+(m.D2     )*(m.exx+m.eyy)
        E1=H1
        E2=((H2+H3)+np.sqrt((H2+H3)**2-4*(H2*H3-2*m.Delta3**2)))/2
        E3=((H2+H3)-np.sqrt((H2+H3)**2-4*(H2*H3-2*m.Delta3**2)))/2

        # This is the top of the kp valence band with strain
        EV=MidFunction(m,np.max([E1,E2,E3],axis=0))


        # How the VB edge moves with strain
        Sigma2=EV-EV0
        # How the CB edge moves with strain
        Sigmac=(m.a1+m.D1)*m.ezz+(m.a2+m.D2)*(m.exx+m.eyy)

        # E0 is the midband energy of the unstrained band
        # Material offsets will be expressed in terms of difference of E0
        m['Eg']=Eg_re + Sigmac-Sigma2
        m['E0-Ev']=Eg_re/2  -Sigma2
        m['Ec-E0']=Eg_re/2  +Sigmac

        # For kp this is the thing to add to Ev before solving Hamiltonian
        # Ev already includes (1) strain shift of VB but I want that in the Hamiltonian,
        # so subtract that out from Ev to get an Ev_raw.
        # Also, the bandedge in the Hamiltonian is not zero, it's max(Delta1+Delta2,Delta1-Delta2,0)
        # So let's set Ev_raw to Ev minus that amount to make sure the bulk solution top energy is Ev
        m['EvOffset']=-EV

        # Similarly, here's the thing to add to Ec before solving kp
        # so that strain shift is included directly into kp Hamiltonian not Ec
        m['EcOffset']=-Sigmac

        if key:
            return m[key]

    def _smcls_band_params(self,m,key):
        log("Using explicit masses from file",'TODO')
        m['eg']=MidFunction(m,2)
        m['hg']=MidFunction(m,2)
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
        m['medos']=m.mez**(1/3)*m.mexy**(2/3)
        m['mhdos']=m.mhz**(1/3)*m.mhxy**(2/3)
        m['cDE']=np.atleast_2d(
            self.vergard('carrier=electron.band=.DE')(m,None))
        m['vDE']=MidFunction(m,np.vstack([
            self.vergard('carrier=hole.band=HH.DE')(m,None),
            self.vergard('carrier=hole.band=LH.DE')(m,None),
            self.vergard('carrier=hole.band=CH.DE')(m,None)]))
        return m[key]

    def _kp_params(self,m,key):
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

    kp_dim={'hole':6,'electron':2}
    """ Dimension of the kp problem, 6 for holes, 2 for electrons."""

    def kp_Cmats(self,m,kx,ky,carrier,kxl=None,kyl=None):
        r""" Returns the kp matrices needed by :class:`~pynitride.carriers.MultibandKP`

        The matrices are in the bases used by
        `Birner <https://www.nextnano.com/downloads/publications/PhD_thesis_Stefan_Birner_TUM_2011_WSIBook.pdf>`_.
        The naming convention for the `C` matrices is given in :ref:`FEM`.

        If `kxl, kyl` are not specified, then `kx, ky` is used for both the left ket wavevector and
        the right ket wavevector.  This is almost always the correct thing to do.
        There are some times when one cares about the matrix element of the kp Hamiltonian between two different
        wavevectors (such as in surface roughness scattering).  In that case, the left ket wavevector can be specified
        separately via `kxl, kyl`.  The resulting matrices may, of course, be non-Hermitian.

        If the artificial spin-splitting (see :class:`Wurtzite`) is zero, the matrices are

        .. math::
            C_{0}=C_{0L}+C_{0D}+C_{0S}

        .. math::
            \begin{equation}
            C_{l}= I_2 \otimes \begin{pmatrix}
                               \cdot     &             \cdot    &       k_x^lN_2^+    \\
                               \cdot     &             \cdot    &       k_y^lN_2^+    \\
                          k_x^lN_2^-    &        k_y^lN_2^-   &         \cdot
            \end{pmatrix}
            \end{equation}

        .. math::
            \begin{equation}
            C_{r}= I_2 \otimes \begin{pmatrix}
                                   \cdot     &             \cdot    &       N_2^-k_x^r \\
                                   \cdot     &             \cdot    &       N_2^-k_y^r \\
                              N_2^+k_x^r    &        N_2^+k_y^r   &         \cdot
            \end{pmatrix}
            \end{equation}

        .. math::
            \begin{equation}
            C_{2}= I_2 \otimes \begin{pmatrix}
                                   M_2^u   &             \cdot    &        \cdot     \\
                                   \cdot     &           M_2^u    &        \cdot     \\
                                   \cdot     &              \cdot   &       L_2^u    \\
            \end{pmatrix}
            \end{equation}

        where :math:`C_{0S}` is the strain matrix from :func:`Wurtzite.kp_strain_mat` and

        .. math::
            \begin{equation}
            C_{0L}= I_2 \otimes \begin{pmatrix}
                k_x^lL_1^uk_x^r+k_y^lM_1^uk_y^r & k_x^lN_1^+k_y^r+k_y^lN_1^-k_x^r&         \cdot\\
                k_y^lN_1^+k_x^r+k_x^lN_1^-k_y^r & k_x^lM_1^uk_x^r+k_y^lL_1^uk_y^r&         \cdot\\
                           \cdot            &            \cdot           &  k_x^lM_3^uk_x^r+k_y^lM_3^uk_y^r
                \end{pmatrix}
            \end{equation}

        .. math::
            \begin{equation}
            C_{0D}=\begin{pmatrix}
                \Delta_1 &   - i\Delta_2 &          \cdot &          \cdot &         \cdot &     \Delta_3 \\
              i\Delta_2 &       \Delta_1 &          \cdot &          \cdot &         \cdot & - i\Delta_3 \\
                     \cdot &            \cdot &          \cdot &    -\Delta_3 &  i\Delta_3 &          \cdot \\
                     \cdot &            \cdot &    -\Delta_3 &     \Delta_1 &  i\Delta_2 &          \cdot \\
                     \cdot &            \cdot & - i\Delta_3 & - i\Delta_2 &    \Delta_1 &          \cdot \\
                \Delta_3 &     i\Delta_3 &          \cdot &          \cdot &         \cdot &          \cdot
            \end{pmatrix}
            \end{equation}

        If the artificial spin-splitting is provided, then the off-diagonal 3x3 blocks of `C_0` are zeroed out, and
        an additional matrix

        .. math::
            \begin{equation}
            SS=\begin{pmatrix}
            ss/2 & \cdot & \cdot & \cdot & \cdot  & \cdot \\
            \cdot & ss/2 & \cdot & \cdot & \cdot  & \cdot \\
            \cdot & \cdot & ss/2 & \cdot & \cdot & \cdot \\
            \cdot & \cdot  & \cdot & -ss/2 & \cdot & \cdot \\
            \cdot & \cdot  & \cdot & \cdot & -ss/2 & \cdot \\
            \cdot & \cdot  & \cdot & \cdot & \cdot & -ss/2
            \end{pmatrix}
            \end{equation}

        is added to :math:`C_0`

        Args:
            m: the mesh
            kx, ky: the wavevector
            carrier: 'electron' or 'hole'
            kxl, kyl: the wavevector of the left ket

        Returns:
            a tuple of (C0,Cl,Cr,C2), where each is an (n mesh.Nn) x (n mesh.Nn) complex matrix,
            `n=2,6` for electrons,holes
        """

        kxr=kx; kyr=ky;
        if kxl is None: kxl=kx
        if kyl is None: kyl=ky
        del kx; del ky;

        if carrier=='electron':
            S1=hbar**2/(2*m.mez[0])
            S2=hbar**2/(2*m.mexy[0])
            strainmat=self.kp_strain_mat(m,exx=m.exx,exy=m.exy,exz=m.exz,eyy=m.eyy,eyz=m.eyz,ezz=m.ezz,carrier=carrier)
            Cmats = []
            for kxl, kyl, kxr, kyr in zip(kxl, kyl, kxr, kyr):
                k2 =  kxl*kxr+kyl*kyr
                C0 = double_mat(MidFunction(m, [[S2 * k2]]), dtype='float')+strainmat
                C0[1, 1] += 5e-6; C0[0, 0] -= 5e-6  # Break degeneracy by 1ueV
                C2 = double_mat(MidFunction(m, [[S1]]), dtype='float')
                Cmats += [[C0, None, None, C2]]
            return Cmats

        if carrier=='hole':
            U=MidFunction(m,hbar**2/(2*m_e))
            O=MidFunction(m,0)
            Atwid=m.A2+m.A4-U; Ahat=m.A1+m.A3-U
            L1=m.A5+Atwid; L2=m.A1-U
            M1=-m.A5+Atwid; M2=Ahat; M3=m.A2-U
            N1p=3*m.A5-Atwid; N1m=-m.A5+Atwid
            N2p=np.sqrt(2)*m.A6-Ahat
            N2m=Ahat
            L1u=L1+U; L2u=L2+U
            M1u=M1+U; M2u=M2+U; M3u=M3+U
            Delta1=m.Delta1;Delta2=m.Delta2;Delta3=m.Delta3
            ss=m.ones_mid*self._spin_splitting

            strainmat=self.kp_strain_mat(m,exx=m.exx,exy=m.exy,exz=m.exz,eyy=m.eyy,eyz=m.eyz,ezz=m.ezz,carrier=carrier)

            Cmats=[]
            for kxl, kyl, kxr, kyr in zip(kxl, kyl, kxr, kyr):
                C0= \
                    double_mat(MidFunction(m, [
                        [kxl*L1u*kxr+kyl*M1u*kyr , kxl*N1p*kyr+kyl*N1m*kxr,         O           ],
                        [kyl*N1p*kxr+kxl*N1m*kyr , kxl*M1u*kxr+kyl*L1u*kyr,         O           ],
                        [           O            ,            O           ,  kxl*M3u*kxr+kyl*M3u*kyr]],dtype='complex')) + \
                    MidFunction(m,[
                         [   Delta1,   -1j*Delta2,          O,          O,         O,     Delta3],
                         [1j*Delta2,       Delta1,          O,          O,         O, -1j*Delta3],
                         [        O,            O,          O,    -Delta3, 1j*Delta3,          O],
                         [        O,            O,    -Delta3,     Delta1, 1j*Delta2,          O],
                         [        O,            O, -1j*Delta3, -1j*Delta2,    Delta1,          O],
                         [   Delta3,    1j*Delta3,          O,          O,         O,          O]],dtype='complex')\
                                                *(self._spin_splitting==0) + \
                    MidFunction(m,[
                         [   Delta1,   -1j*Delta2,          O,          O,         O,          O],
                         [1j*Delta2,       Delta1,          O,          O,         O,          O],
                         [        O,            O,          O,          O,         O,          O],
                         [        O,            O,          O,     Delta1, 1j*Delta2,          O],
                         [        O,            O,          O, -1j*Delta2,    Delta1,          O],
                         [        O,            O,          O,          O,         O,          O]],dtype='complex')\
                                                *(self._spin_splitting!=0) + \
                    MidFunction(m,[
                         [     ss/2,            O,          O,          O,         O,          O],
                         [        O,         ss/2,          O,          O,         O,          O],
                         [        O,            O,       ss/2,          O,         O,          O],
                         [        O,            O,          O,      -ss/2,         O,          O],
                         [        O,            O,          O,          O,     -ss/2,          O],
                         [        O,            O,          O,          O,         O,      -ss/2]],dtype='complex') + \
                    strainmat

                Cl= m.ztrans * \
                    double_mat(MidFunction(m, [
                        [           O     ,             O    ,       kxl*N2p     ],
                        [           O     ,             O    ,       kyl*N2p     ],
                        [      kxl*N2m    ,        kyl*N2m   ,         O        ]]))
                Cr= m.ztrans * \
                    double_mat(MidFunction(m, [
                        [           O     ,             O    ,       N2m*kxr     ],
                        [           O     ,             O    ,       N2m*kyr     ],
                        [      N2p*kxr    ,        N2p*kyr   ,         O        ]]))

                C2= \
                    double_mat(MidFunction(m, [
                        [           M2u   ,             O    ,        O     ],
                        [           O     ,           M2u    ,        O     ],
                        [           O     ,              O   ,       L2u    ]]))
                Cmats+=[[C0,Cl,Cr,C2]]
            return Cmats

    def kp_strain_mat(self,m,carrier,**strains):
        r""" Returns the strain deformation matrix
        by the means noted on pg 2496 of `CC96 <https://doi.org/10.1103/PhysRevB.54.2491>`_.

        See examples/wzstrainterms.ipynb to show that this is consistent.

        For conduction band

        .. math::

            \begin{pmatrix}
                a_{c2} e_{xx} + a_{c2} e_{yy} + a_{c1} e_{zz} & 0 \\
                0 & a_{c2} e_{xx} + a_{c2} e_{yy} + a_{c1} e_{zz} \\
            \end{pmatrix}


        .. math::

            \begin{gather}
                a_{c1}=a_1+D_1,\quad a_{c2}=a_2+D_2
            \end{gather}


        For valence band, the 6x6 strain matrix is a block diagonal of two 3x3 blocks

        .. math::

            \begin{pmatrix}
                l_1 e_{xx} +m_1 e_{yy} +m_2 e_{zz} & n_1 e_{xy} & n_2 e_{xz}\\
                n_1 e_{xy} & m_1 e_{xx} +l_1 e_{yy} +m_2 e_{zz} & n_2 e_{yz}\\
                n_2 e_{xz} & n_2 e_{yz} &  m_3 e_{xx} +m_3 e_{yy} +l_2 e_{zz}\\
            \end{pmatrix}

        .. math::

            \begin{gather}
                l_1=D_2+D_4+D_5, \quad l_2=D_1\\
                m_1=D_2+D_4-D_5, \quad m_2=D_1+D_3 \quad m_3=D_2\\
                n_1=2 D_5 \quad n_2=\sqrt{2} D_6
            \end{gather}


        Args:
            m: the mesh
            strains: optional dictionary of strain fuctions ('exx', 'exy' etc).  If not provided, these will be taken
                from the mesh

        Returns:
            a (n mesh.Nn) x (n mesh.Nn) complex matrix, `n=2,6` for electrons,holes
        """
        dtype='float'
        s=strains.copy()
        for sij in ['exx','exy','exz','eyy','eyz','ezz']:
            if sij not in s: s[sij]=m[sij]
            if s[sij].dtype==complex:
                dtype='complex'

        if carrier=='electron':
            ac1=m.a1+m.D1; ac2=m.a2+m.D2
            return double_mat(MidFunction(m,[[ac2*s['exx']+ac2*s['eyy']+ac1*s['ezz']]],dtype=dtype),dtype=dtype)
        if carrier=='hole':
            l1=m.D2+m.D4+m.D5; l2=m.D1
            m1=m.D2+m.D4-m.D5; m2=m.D1+m.D3; m3=m.D2
            n1=2*m.D5; n2=np.sqrt(2)*m.D6
            return double_mat(MidFunction(m, [
                [l1*s['exx']+m1*s['eyy']+m2*s['ezz'],     n1*s['exy'],                n2*s['exz']],
                [       n1*s['exy'],      m1*s['exx']+l1*s['eyy']+m2*s['ezz'],        n2*s['eyz']],
                [       n2*s['exz'],                      n2*s['eyz'],         m3*s['exx']+m3*s['eyy']+l2*s['ezz']]],\
                dtype='complex'))

    def ec_Cmats(self,m,q):
        r""" Elastic continuum matrices.

        .. math::
            \begin{equation}
            C_0=\begin{pmatrix}
                C_{11}  &  \cdot & \cdot \\
                \cdot   &  (C_{11}-C_{12})/2 & \cdot \\
                \cdot & \cdot & C_{44}
            \end{pmatrix}
            \end{equation}

        .. math::
            \begin{equation}
            C_l=\begin{pmatrix}
                \cdot  &  \cdot & C_{13} \\
                \cdot   &  \cdot & \cdot \\
                C_{44} & \cdot & \cdot
            \end{pmatrix}
            \end{equation}

        .. math::
            \begin{equation}
            C_r=\begin{pmatrix}
                \cdot  &  \cdot & C_{44} \\
                \cdot   &  \cdot & \cdot \\
                C_{13} & \cdot & \cdot
            \end{pmatrix}
            \end{equation}

        .. math::
            \begin{equation}
            C_2=\begin{pmatrix}
                C_{44}  &  \cdot & \cdot \\
                \cdot   &  C_{44} & \cdot \\
                \cdot & \cdot & C_{33}
            \end{pmatrix}
            \end{equation}

        Args:
            m: the mesh
            q: the in-plane wavevector

        Returns:
            a tuple of (C0,Cl,Cr,C2), where each is an (3 mesh.Nn) x (3 mesh.Nn) complex matrix

        """
        q=np.reshape(q,(len(q),1,1,1))
        O=MidFunction(m,0)
        C0=MidFunction(m,q**2*(np.array([
            [    m.C11,               O,         O],
            [        O, (m.C11-m.C12)/2,         O],
            [        O,               O,     m.C44]])))
        Cl=(m.ztrans*MidFunction(m,q*np.array([
            [        O,               O,     m.C13],
            [        O,               O,         O],
            [    m.C44,               O,         O]])))
        Cr=(m.ztrans*MidFunction(m,q*np.array([
            [        O,               O,     m.C44],
            [        O,               O,         O],
            [    m.C13,               O,         O]])))
        C2=0*q+np.array([
            [    m.C44,               O,         O],
            [        O,           m.C44,         O],
            [        O,               O,     m.C33]])
        return [[C0[i],Cl[i],Cr[i],C2[i]] for i in range(len(q))]

    def ec_CmatsXZ(self,m,q):
        r""" Like :func:`Wurtzite.ec_Cmats` but only the 2x2 XZ matrices."""

        q=np.reshape(q,(len(q),1,1,1))
        O=MidFunction(m,0)
        C0=MidFunction(m,q**2*(np.array([
            [    m.C11,         O],
            [        O,     m.C44]])))
        Cl=(m.ztrans*MidFunction(m,q*np.array([
            [        O,     m.C13],
            [    m.C44,         O]])))
        Cr=(m.ztrans*MidFunction(m,q*np.array([
            [        O,     m.C44],
            [    m.C13,         O]])))
        C2=0*q+np.array([
            [    m.C44,         O],
            [        O,     m.C33]])
        return [[C0[i],Cl[i],Cr[i],C2[i]] for i in range(len(q))]

    def ec_CmatsY(self,m,q):
        r""" Like :func:`Wurtzite.ec_Cmats` but only the 1x1 Y (center) matrices."""
        q=np.reshape(q,(len(q),1,1,1))
        O=MidFunction(m,0)
        C0=MidFunction(m,q**2*(np.array([[(m.C11-m.C12)/2]])))
        Cl=0*q+MidFunction(m,np.array([[O]]))
        Cr=0*q+MidFunction(m,np.array([[O]]))
        C2=0*q+np.array([[m.C44]])
        return [[C0[i],Cl[i],Cr[i],C2[i]] for i in range(len(q))]

    def strain_to(self,m,straincond={}):
        """ Strains the material to the specified condition.

        The `straincond` dict can contain a key 'a', in which case this will be the in-plane lattice constant.
        Or the `straincond` dict can contain separate keys `ax`, `ay` for two lattice constants.

        In addition to the above choice, a `zcond` entry can indicate how the stress/strain in the z-direction
        should be handled: 'free' is the typical pseudomorphic condition (for wurtzite,
        :math:`e_{zz}=-C_{13}/C_{33}(e_{xx}+e_{yy})`).

        This function populates all the `eij` variables onto the mesh, setting zero for the shear components.

        Args:
            m: the mesh
            straincond: see above

        """
        a0=self.vergard('conditions=relaxed.lattice.a')(m,None)

        if 'a' in straincond:
            ax=ay=straincond['a']
        else:
            if 'ax' in straincond:
                ax=straincond['ax']
            else: raise NotImplementedError
            if 'ay' in straincond:
                ay=straincond['ay']
            else: raise NotImplementedError
        m['exx']=(ax-a0)/a0
        m['eyy']=(ay-a0)/a0

        zcond=straincond.get('zcond','free')
        if   zcond=='free':
            m['ezz']=-m.C13/m.C33*(m['exx']+m['eyy'])
        elif zcond=='fixed':
            m['ezz']=0
        else:
            raise Exception("Unrecognized strain z-condition")

        m['exy']=m['eyz']=m['exz']=MidFunction(m,0)

    def bulk_lattice_condition(self,m):
        """ The natural lattice constant of the bottommost point"""
        pos=-1 if (m.ztrans == -1 ) else 0
        a0=self.vergard('conditions=relaxed.lattice.a')(m,None)[pos]
        return {'a':a0}



class AlGaInN(Wurtzite):
    def __init__(self):
        self.vergardbasis={
            'x': 'AlN',
            'y': 'InN',
            None: 'GaN',
        }

        super().__init__()

        self._defaults.update({
            'x': 0,
            'y': 0,
            'gotz': True
        })
        self.append_dopants(['Si','Mg','Deep'])

        self._attrs['MgAcceptorE']

class AlGaN(Wurtzite):
    def __init__(self,spin_splitting=0):
        self.vergardbasis={
            'x': 'AlN',
            None: 'GaN',
        }
        self.name="AlGaN"

        super().__init__(spin_splitting=spin_splitting)

        self._defaults.update({
            'x': 0
        })
        self.append_dopants(['Si','Mg','DeepDonor','DeepAcceptor'])

class Insulator(MaterialSystem):
    def __init__(self, name):
        self.vergardbasis={None: name}
        self.name=name
        super().__init__()
        self._updates.update({
            'temperature': [self._bandedge_params],
        })
        self.append_dopants([])

    def _bandedge_params(self,m,key):
        Eg0=self.vergard('conditions=relaxed.varshni.Eg0')(m,None)
        alpha=self.vergard('conditions=relaxed.varshni.alpha')(m,None)
        beta=self.vergard('conditions=relaxed.varshni.beta')(m,None)
        Eg_re=Eg0-alpha*m.T**2/(m.T+beta)

        m['Eg']=Eg_re
        m['E0-Ev']=Eg_re/2
        m['Ec-E0']=Eg_re/2

        return m[key]


