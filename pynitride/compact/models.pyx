cimport cython
cimport numpy as cnp
import numpy as np
from libc.math cimport pow, exp, log, abs, log10
from functools import wraps
cnp.import_array()

from pynitride.paramdb import ParamDB
cdef double k,e
k,e=ParamDB(units="si").get_constants("k,e")

cdef class GaNHEMT_iMVGS:

    cdef double _Vth, W, Cinv_vxo, VT0, alpha, SS, delta, VDsats, beta, eta, Gleak, n, Rs, Rd

    r""" intrinsic MVGS GaN HEMT

    The intrinsic (ie no access resistance) model in :ref:`U. Radhakrishna's Thesis <http://www.mit.edu/~ujwal/MSthesis.pdf>`_

    :param T: temperature in Kelvin
    :param W: device width in meters
    :param Cinv_vxo: transconductance per width in Siemens per meter
    :param VT: threshold voltage in Volts
    :param alpha: threshold shift between strong and weak inversion is ``alpha`` times thermal voltage
    :param SS: subthreshold swing in Volts per decade
    :param Yleak: Leakage conductance in Siemens
    """
    @cython.cdivision(True)
    def __init__(GaNHEMT_iMVGS self, double T=300,
            double W=1000e-6, double Cinv_vxo=.11e3,
            double VT0=-4.2, double alpha=3.5, double SS=90e-3, double delta=0,
            double VDsats=4, double beta=2, double eta=.1, double Gleak=1e-14,
            double Rs=0,double Rd=0):

        self._Vth=k*T/e
        self.W=W
        self.Cinv_vxo=Cinv_vxo
        self.VT0=VT0
        self.alpha=alpha
        self.SS=SS
        self.delta=delta
        self.VDsats=VDsats
        self.beta=beta
        self.eta=eta
        self.Gleak=Gleak
        self.n=SS/(self._Vth*log(10))
        self.Rs=Rs
        self.Rd=Rd



    # Eq 4.4
    cdef double VT(GaNHEMT_iMVGS self, double VDS):
        return self.VT0-self.delta*VDS

    # Eq 4.5
    @cython.cdivision(True)
    cdef double Ff(GaNHEMT_iMVGS self, double VDS, double VGS):
        if self.alpha==0: return 0
        return 1/(1+exp((VGS-(self.VT(VDS)-self.alpha*self._Vth/2))/(self.alpha*self._Vth)))

    # Eq 4.6
    @cython.cdivision(True)
    cdef double Fsat(GaNHEMT_iMVGS self, double VDS, double VGS) except? -1:
        if VDS<0:
            raise Exception("What you tryna do to me?  ",VDS)
            return -1
        VDsat=self.VDsats*(1-self.Ff(VDS,VGS))+self._Vth*self.Ff(VDS,VGS)
        return (VDS/VDsat)/(1+(VDS/VDsat)**self.beta)**(1/self.beta)

    # Drain current
    cdef double ID(GaNHEMT_iMVGS self,double VDS, double VGS):

        cdef double VGsi, VDSi, IDprev
        VGSi=VGS
        VDSi=VDS
        IDprev=None

        while True:
            # Eq 4.2, 4.7 + plus a leakage VDS*1e-14
            ID0=self.W*self.Cinv_vxo*self.Fsat(VDSi,VGSi)*self.n*self._Vth\
                *log(1+exp((VGSi-(self.VT(VDSi)-self.alpha*self._Vth*self.Ff(VDSi,VGSi)))/(self.n*self._Vth)))\
                +VDSi*self.Gleak

            # Eq 4.10 with theta_v=0
            ID=ID0/(1+self.eta*ID0*VDSi)

            # If there is no added resistance
            if not (self.Rs or self.Rd):
                break
            if IDprev is not None:
                if abs(log10((ID+1e-14)/(IDprev+1e-14)))<.0001:
                    break
            IDprev=ID
            #print(VGSi)
            VGSi=VGSi+.2*(VGS-ID*self.Rs-VGSi)
            VDSi=VDSi+.2*(np.clip(VDS-ID*(self.Rs+self.Rd),1e-8,VDS)-VDSi)
        return ID
