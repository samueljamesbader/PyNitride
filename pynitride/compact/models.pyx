cimport cython
cimport numpy as cnp
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from libc.math cimport pow, exp, log, abs, log10
from functools import wraps
cnp.import_array()

from pynitride.paramdb import ParamDB
cdef double k,e
k,e=ParamDB(units="si").get_constants("k,e")
#print(k)

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
    cpdef double Ff(GaNHEMT_iMVGS self, double VDS, double VGS):
        cdef double exparg
        if self.alpha==0: return 0
        else:
            exparg=(VGS-(self.VT(VDS)-self.alpha*self._Vth/2))/(self.alpha*self._Vth)
            if exparg>15: return 0
            if exparg<-15: return 1
            return 1/(1+exp(exparg))

    # Eq 4.6
    @cython.cdivision(True)
    cpdef double Fsat(GaNHEMT_iMVGS self, double VDS, double VGS) except? -1:
        if VDS<0:
            #raise Exception("What you tryna do to me?  ",VDS)
            return -1
        VDsat=self.VDsats*(1-self.Ff(VDS,VGS))+self._Vth*self.Ff(VDS,VGS)
        return (VDS/VDsat)/(1+(VDS/VDsat)**self.beta)**(1/self.beta)

    # Drain current
    @cython.cdivision(True)
    cpdef double _ID(GaNHEMT_iMVGS self,double VDS, double VGS):

        cdef double VGsi, VDSi, IDprev, ID
        VGSi=VGS
        VDSi=VDS
        IDprev=-1

        while True:
            # Eq 4.2, 4.7 + plus a leakage VDS*1e-14

            exparg=(VGSi-(self.VT(VDSi)-self.alpha*self._Vth*self.Ff(VDSi,VGSi)))/(self.n*self._Vth)
            if exparg<15:
                ID0=self.W*self.Cinv_vxo*self.Fsat(VDSi,VGSi)*self.n*self._Vth\
                    *log(1+exp(exparg))\
                    +VDSi*self.Gleak
            else:
                ID0=self.W*self.Cinv_vxo*self.Fsat(VDSi,VGSi)*self.n*self._Vth \
                    *(exparg) \
                    +VDSi*self.Gleak

            # Eq 4.10 with theta_v=0
            ID=ID0/(1+self.eta*ID0*VDSi)

            # If there is no added resistance
            if not (self.Rs or self.Rd):
                break

            assert False, "I didn't Cythonize the R-loop"
            if IDprev>0:
                if abs(log10((ID+1e-14)/(IDprev+1e-14)))<.0001:
                    break
            IDprev=ID
            VGSi=VGSi+.2*(VGS-ID*self.Rs-VGSi)
            print("USING PYTHON")
            VDSi=VDSi+.2*(np.clip(VDS-ID*(self.Rs+self.Rd),1e-8,VDS)-VDSi)
        return ID

    def ID(GaNHEMT_iMVGS self, VD, VG):
        cdef int j
        cdef double[:] iarr, vdarr, vgarr
        cdef double vg, vd

        VDgrid,VGgrid=np.meshgrid(VD,VG)
        I = np.empty_like(VDgrid)
        it = np.nditer([VDgrid,VGgrid,I], flags=['external_loop','buffered'],
                       op_flags=[['readwrite'], ['readwrite'], ['readwrite']])
        for vdarr,vgarr,iarr in it:
            for j in range(vdarr.shape[0]):
                vd=vdarr[j]
                vg=vgarr[j]
                iarr[j]=self._ID(vdarr[j],vgarr[j])
        return I


cdef struct PiecewiseLinearVI:
    double I_trans
    double R_low
    double R_high
    double V_off_high

cpdef enum Direction:
    FORWARD = 0
    BACKWARD = 1

cdef class VO2Res:

    cdef public double I_IMT, I_MIT, V_IMT, V_MIT, R_ins, V_met, R_met, I_smooth
    cdef PiecewiseLinearVI[2] VIs

    @cython.cdivision(True)
    def __init__(VO2Res self, double I_IMT=.01e-3, double V_IMT=1, double I_MIT=-1, double V_MIT=.5, double R_met=0):
        # I_IMT: Current threshold [I]
        # V_IMT: Voltage threshold [V]
        # V_MIT=Voltage required to sustain metallic conduction [V]
        # R_met Resistance in the metallic state [Ohm]

        self.I_IMT=I_IMT
        self.I_MIT=I_MIT if I_MIT>0 else I_IMT*.75
        self.V_IMT=V_IMT
        self.V_MIT=V_MIT
        self.R_ins=V_IMT/I_IMT
        self.V_met=V_MIT-self.I_MIT*R_met
        self.R_met=R_met

        self.VIs[<int> Direction.FORWARD]=\
            PiecewiseLinearVI(I_trans=self.I_IMT,R_low=self.R_ins,R_high=self.R_met,V_off_high=self.V_met)
        self.VIs[<int> Direction.BACKWARD]=\
            PiecewiseLinearVI(I_trans=self.I_MIT,R_low=self.R_ins,R_high=self.R_met,V_off_high=self.V_met)

    @cython.boundscheck(False)
    def V(VO2Res self, I, Direction direc):
        cdef int j
        cdef double[:] iarr, varr
        cdef PiecewiseLinearVI vi

        vi=self.VIs[<int>direc]
        #print(vi)

        V=np.empty_like(I)
        it = np.nditer([I, V], flags=['external_loop','buffered'],
                       op_flags=[['readwrite'], ['readwrite']],
                       op_dtypes=['float64', 'float64'])
        for iarr,varr in it:
            for j in range(iarr.shape[0]):
                i=iarr[j]
                if i<vi.I_trans:
                    varr[j]=i*vi.R_low
                else:
                    varr[j]=i*vi.R_high+vi.V_off_high
        return V


# cdef double bounded_newton(double (*f)(double,list),
#                            double (*fp)(double,list),
#                            double xmin, double xmax, double x0,
#                            double xtol, double maxiter, double sor=0):
#     cdef int i=0
#     cdef double x=x0, y, s, dx
#
#     for i in range(maxiter):
#         y=f(x)
#         s=fp(x)
#         dx=-y/s
#
#         x=x+sor*dx
#
#         if xi<xmin:
#             xi=xmin
#         if xi>xmax:
#             xi=xmax
#         if dx<xtol:
#             return xi
#
#     return -1



cdef class HyperFET:

    cdef GaNHEMT_iMVGS hemt
    cdef VO2Res vo2

    def __init__(HyperFET self, GaNHEMT_iMVGS hemt, VO2Res vo2):
        self.hemt=hemt
        self.vo2=vo2

    cdef double _I_low(HyperFET self, double VD, double VG, Direction direc) except? -1:
        rVI=self.vo2.VIs[<int>direc]

        imax_r=rVI.I_trans
        imax_vd=VD/rVI.R_low
        imax=min(imax_r,imax_vd)*.9999

        def ierr( double log_i_try, HyperFET self, double VD, double VG):
            cdef double i_try, v_r, i_calc
            i_try=exp(log_i_try)
            v_r=i_try*rVI.R_low
            i_calc=self.hemt._ID(VD-v_r,VG-v_r)
            return log(i_calc)-log_i_try

        logimin=-50
        logimax=log(imax)
        if(ierr(logimin,self,VD,VG)<0):
            raise Exception("Min too high")
            return -1
        if(ierr(logimax,self,VD,VG)>0):
            return -1

        x=brentq(ierr,logimin,logimax,args=(self,VD,VG),xtol=1e-5,rtol=1e-5)
        if abs(ierr(x,self,VD,VG))< .001:
            return exp(x)
        else: return -1

    cdef double _I_high(HyperFET self, double VD, double VG, Direction direc):
        rVI=self.vo2.VIs[<int>direc]

        imin=rVI.I_trans
        imax=(VD-rVI.V_off_high)/rVI.R_high

        def ierr( double log_i_try, HyperFET self, double VD, double VG):
            cdef double i_try, v_r
            i_try=exp(log_i_try)
            v_r=i_try*rVI.R_high+rVI.V_off_high
            i_calc=self.hemt._ID(VD-v_r,VG-v_r)
            return log(i_calc)-log_i_try

        logimin=log(imin)
        logimax=log(imax)
        if(ierr(logimin,self,VD,VG)<0):
            return -1
        if(ierr(logimax,self,VD,VG)>0):
            raise Exception("Max too low")
            return -1

        x=brentq(ierr,logimin,logimax,args=(self,VD,VG),xtol=1e-5,rtol=1e-5)
        if abs(ierr(x,self,VD,VG))< .001:
            return exp(x)
        else: return -1

    def I(HyperFET self, VD, VG, Direction direc):
        cdef int j
        cdef double[:] iarr, vdarr, vgarr
        cdef double vd_prev, vg_prev, i_prev, i_trans, vd, vg

        i_trans=self.vo2.VIs[<int>direc].I_trans

        i_prev=i_trans
        VDgrid,VGgrid=np.meshgrid(VD,VG)
        I = np.empty_like(VDgrid)
        it = np.nditer([VDgrid,VGgrid,I], flags=['external_loop','buffered'],
                       op_flags=[['readwrite'], ['readwrite'], ['readwrite']])
        for vdarr,vgarr,iarr in it:
            for j in range(vdarr.shape[0]):
                vd=vdarr[j]
                vg=vgarr[j]

                if direc==Direction.FORWARD:
                    skip_low=(vd>vd_prev and vg>vg_prev and i_prev>i_trans)
                    vg_prev=vg
                    vd_prev=vd
                    if not skip_low:
                        i=self._I_low(vd,vg,Direction.FORWARD)
                        if i>0:
                            i_prev=iarr[j]=i
                            continue
                    i=self._I_high(vd,vg,Direction.FORWARD)
                    if i>0:
                        i_prev=iarr[j]=i
                    else:
                        iarr[j]=np.NaN
                        #raise Exception("Solution not found")
                else:
                    skip_high=(vd<vd_prev and vg<vg_prev and i_prev<i_trans)
                    if not skip_high:
                        i=self._I_high(vd,vg,Direction.BACKWARD)
                        if i>0:
                            i_prev=iarr[j]=i
                            continue
                    i=self._I_low(vd,vg,Direction.BACKWARD)
                    if i>0:
                        i_prev=iarr[j]=i
                    else:
                        iarr[j]=np.NaN
                        #raise Exception("Solution not found")
        return I

    def I_double(HyperFET self, VD, VG):
        def rev(arr):
            if len(arr.shape)>0:
                arr=np.flipud(arr)
            if len(arr.shape)>1:
                arr=np.fliplr(arr)
            return arr

        If=self.I(VD,VG,Direction.FORWARD)
        Ib=rev(self.I(rev(VD),rev(VG),Direction.BACKWARD))
        return If,Ib
