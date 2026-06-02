import numpy as np
import matplotlib.pyplot as plt

from pynitride.core.reciprocal_mesh import RMesh1D
from pynitride.physics.material import AlInGaN
from pynitride.core.mesh import Mesh, MaterialBlock, UniformLayer
from pynitride import pmdb, nm, hbar, m_e, meV
from pynitride.physics.carriers import MultibandKP
from pynitride.physics.strain import Pseudomorphic
from pynitride.physics.thermal import ConstantT

def get_III_N_masses():
    """Computes and prints small-k and large-k limit effective masses based on database Rashba-Sheka-Pikus parameters."""
    # Table I of Chuan and Chang 1996, doi.org//10.1103/PhysRevB.54.2491
    for matname,kwargs in [('GaN',dict(x=0,y=0)),('AlN',dict(x=1,y=0)),('InN',dict(x=0,y=1)),]:

        # Material parameters
        A1 = pmdb[f'{matname}.kp.A1']
        A2 = pmdb[f'{matname}.kp.A2']
        A3 = pmdb[f'{matname}.kp.A3']
        A4 = pmdb[f'{matname}.kp.A4']
        A5 = pmdb[f'{matname}.kp.A5']
        DeltaCR = pmdb[f'{matname}.kp.DeltaCR']
        DeltaSO = pmdb[f'{matname}.kp.DeltaSO']

        # Crystal-field and spin-orbit splittings
        Delta1 = DeltaCR
        Delta2 = DeltaSO / 3.0
        Delta3 = DeltaSO / 3.0  # In the quasi-cubic approximation, Delta3 = Delta2 = DeltaSO/3

        # Strain-related terms (set to zero if unstrained)
        theta_eps = 0.0   # θ_ε
        lambda_eps = 0.0  # λ_ε

        # Valence band edge energies at k = 0
        E1_0 = Delta1 + Delta2 + theta_eps + lambda_eps

        half = (Delta1 - Delta2 + theta_eps) / 2.0
        radical = np.sqrt(half**2 + 2.0 * Delta3**2)

        E2_0 = half + lambda_eps + radical
        E3_0 = half + lambda_eps - radical

        # Effective masses near the band edge (k -> 0)
        # Returned as m0/m^z and m0/m^t (inverse effective masses in units of m0)

        # E1: Heavy-hole (HH) band
        m0_over_mz_HH = -(A1 + A3)
        m0_over_mt_HH = -(A2 + A4)

        # E2: Light-hole (LH) band
        frac_E2 = (E2_0 - lambda_eps) / (E2_0 - E3_0)
        m0_over_mz_LH = -(A1 + frac_E2 * A3)
        m0_over_mt_LH = -(A2 + frac_E2 * A4)

        # E3: Crystal-field split-off hole (CH) band
        frac_E3 = (E3_0 - lambda_eps) / (E3_0 - E2_0)
        m0_over_mz_CH = -(A1 + frac_E3 * A3)
        m0_over_mt_CH = -(A2 + frac_E3 * A4)

        # To get the actual effective masses (in units of m0):
        mz_HH = 1.0 / m0_over_mz_HH
        mt_HH = 1.0 / m0_over_mt_HH
        mz_LH = 1.0 / m0_over_mz_LH
        mt_LH = 1.0 / m0_over_mt_LH
        mz_CH = 1.0 / m0_over_mz_CH
        mt_CH = 1.0 / m0_over_mt_CH
        # Effective masses far from the band edge (k large)
        m0_over_mz_HH_far = -(A1 + A3)
        m0_over_mt_HH_far = -(A2 + A4 - A5)
        m0_over_mz_LH_far = -(A1 + A3)
        m0_over_mt_LH_far = -(A2 + A4 + A5)
        m0_over_mz_CH_far = -A1
        m0_over_mt_CH_far = -A2

        mz_HH_far = 1.0 / m0_over_mz_HH_far
        mt_HH_far = 1.0 / m0_over_mt_HH_far
        mz_LH_far = 1.0 / m0_over_mz_LH_far
        mt_LH_far = 1.0 / m0_over_mt_LH_far
        mz_CH_far = 1.0 / m0_over_mz_CH_far
        mt_CH_far = 1.0 / m0_over_mt_CH_far

        # Print values
        Ev=max(E1_0,E2_0,E3_0)
        print(f"{matname}:")
        print(f"  E1-E1 (HH) at k=0: {Ev-E1_0:.3f} eV")
        print(f"  E1-E2 (LH) at k=0: {Ev-E2_0:.3f} eV")
        print(f"  E1-E3 (CH) at k=0: {Ev-E3_0:.3f} eV")
        print(f"  E1 (HH) near the edge: m_z = {mz_HH:.2f} m0, m_t = {mt_HH:.2f} m0")
        print(f"  E2 (LH) near the edge: m_z = {mz_LH:.2f} m0, m_t = {mt_LH:.2f} m0")
        print(f"  E3 (CH) near the edge: m_z = {mz_CH:.2f} m0, m_t = {mt_CH:.2f} m0")
        print(f"  E1 (HH) far from edge: m_z = {mz_HH_far:.2f} m0, m_t = {mt_HH_far:.2f} m0")
        print(f"  E2 (LH) far from edge: m_z = {mz_LH_far:.2f} m0, m_t = {mt_LH_far:.2f} m0")
        print(f"  E3 (CH) far from edge: m_z = {mz_CH_far:.2f} m0, m_t = {mt_CH_far:.2f} m0")

def plot_III_N_masses():
    """Plots k.p bands (dots) versus parabolic approximations based on database masses (solid)"""
    rx=RMesh1D(np.linspace(0,1/(1*nm)))
    kz=np.linspace(0,1/(1*nm))

    for matname,kwargs in [('GaN',dict(x=0,y=0)),
                           ('AlN',dict(x=1,y=0)),
                           ('InN',dict(x=0,y=1)),
                           ]:
        mbkp_x=MultibandKP(AlInGaN().bulk(**kwargs),rx,num_eigenvalues=6,carriers=['hole'])
        mbkp_z=MultibandKP(AlInGaN().bulk(**kwargs),RMesh1D([0,1]),num_eigenvalues=6,carriers=['hole'])
        E_alongx,_=mbkp_x.solve_point_as_bulk(None,kz=0)
        E_alongy=np.array([mbkp_x.solve_point_as_bulk(None,kz=kzi)[0][0,:]
                                for kzi in kz])

        plt.figure(matname)
        plt.ylabel("Energy [meV]")
        plt.xlabel("k [1/nm]")
        # k.p
        plt.plot(rx.absk1/(1/nm),E_alongx/meV,'.')
        plt.plot(-kz/(1/nm),E_alongy/meV,'.')
        # parabolic
        for sb in pmdb[f'{matname}.hole.band']:
            mx=pmdb[f'{matname}.hole.{sb}.mxys']
            mz=pmdb[f'{matname}.hole.{sb}.mzs']
            de=pmdb[f'{matname}.hole.{sb}.DE']
            plt.plot(rx.absk1/(1/nm),-hbar**2*(rx.absk1)**2/(2*mx)-de/meV)
            plt.plot(-kz/(1/nm),-hbar**2*(kz)**2/(2*mz)-de/meV)
    plt.show()

if __name__=="__main__":
    get_III_N_masses()
    plot_III_N_masses()