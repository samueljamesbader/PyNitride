r""" Contains :math:`k\cdot p` models to generate band structures."""

import numpy as np
import scipy.linalg as la
from pynitride.bandstuct.depends import varshni

def kp_6x6(material, kvecs, strainvec, spin_orbit=True):
    r""" Solves the 6x6 + 2x2 :math:`k\cdot p` Hamiltonian for a wurtzite crystal.

    The Hamiltonian can be found in `(Ren et al 97) <http://dx.doi.org/10.1063/1.123461>`_.  This function adds one
    extra constant term, which is the maximum of :math:`(\Delta_1 + \Delta_2,\Delta_1-\Delta_2,0)` which just shifts the
    energies to ensure that the valence band maximum for the unstrained material is 0eV.  Then the conduction band
    minimum for the unstrained material is :math:`E_g`, obtained from the Varshni model

    :param material: the :py:class:`~pynitride.paramdb.Material` under study
    :param kvecs: an iterable of kvectors, each vector being a three-element sequence :math:`k_x,k_y,k_z`.  Typically,
        this input comes from the output of a call to :py:func:`~pynitride.bandstruct.reciprocal.generate_path`.
    :param strainvec: a vector of the :math:`e_x,e_y,e_z` strain components
    :param spin_orbit: whether to include spin-orbit splitting (default True)
    :return: a 2D Numpy array, each row being the eight band energies ordered ascending at a given ``kvec``
    """
    assert material['crystal']=='wurtzite'

    # Get valence k.p band parameters
    A1,A2,A3,A4,A5,A6,A7,D1,D2,D3,D4,D5,D6,a1,a2,DeltaCR,DeltaSO=\
        [material['kp.'+var] for var in \
         "A1,A2,A3,A4,A5,A6,A7,D1,D2,D3,D4,D5,D6,a1,a2,DeltaCR,DeltaSO".split(',')]
    mezs,mexys=[material['electron.'+var] for var in "mzs,mxys".split(',')]
    if not spin_orbit: DeltaSO=0

    # Get unstrained gap
    Eg=varshni(material)

    # a1/2 is movement of conduction band relative to CH
    # ac1/2 is absolute movement of conduction band
    acz=a1+D1
    act=a2+D2

    # VM2003 pg top left of 3679
    Delta1=DeltaCR
    Delta2=DeltaSO/3
    Delta3=DeltaSO/3


    def for_single_k(k):
        # Break up k ane e for convenience
        kx, ky, kz = k
        exx, eyy, ezz = strainvec

        # Ren Eq 2b
        Delta=np.sqrt(2)*Delta3
        Lambda = (A1 * kz ** 2 + A2 * (kx ** 2 + ky ** 2)) + D1 * ezz + D2 * (exx + eyy)
        Theta = (A3 * kz ** 2 + A4 * (kx ** 2 + ky ** 2)) + D3 * ezz + D4 * (exx + eyy)
        F = Delta1 + Delta2 + Lambda + Theta
        G = Delta1 - Delta2 + Lambda + Theta
        K = A5 * (kx + 1j * ky) ** 2
        H = (1j * A6 * kz - A7) * (kx + 1j * ky)
        I = (1j * A6 * kz + A7) * (kx + 1j * ky)

        # Ren Eq 2a
        hmat = np.matrix([
            [F, 0, -H.conjugate(), 0, K.conjugate(), 0],
            [0, G, Delta, -H.conjugate(), 0, K.conjugate()],
            [-H, Delta, Lambda, 0, I.conjugate(), 0],
            [0, -H, 0, Lambda, Delta, I.conjugate()],
            [K, 0, I, Delta, G, 0],
            [0, K, 0, I, 0, F]])

        # Shift so top of unstrained VB is zero
        hmat-=max(Delta1 + Delta2,Delta1-Delta2,0) * np.eye(6)

        # Solve valence band
        VB=la.eigvalsh(hmat)

        # Combine with conduction band
        hbar=material._pmdb.get_constants("hbar")
        T=material._pmdb["T"]
        CB=Eg+np.array([
            hbar ** 2 * (kx ** 2 + ky ** 2) / (2 * mexys)\
            + hbar ** 2 * kz ** 2 / (2 * mezs)\
            + acz * ezz + act * (exx + eyy)])
        return np.concatenate([VB,CB])

    return np.vstack([for_single_k(k) for k in kvecs])


