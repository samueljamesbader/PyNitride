import numpy as np
import scipy.linalg as la

def varshni(material):
    T=material._pmdb["T"]
    return material["varshni.Eg0"]-material["varshni.alpha"]*T**2/(T+material["varshni.beta"])


def kp_bandstructure(material,kvecs,strainvec, spin_orbit=True):
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


