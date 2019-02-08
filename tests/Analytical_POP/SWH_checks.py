import numpy as np
from pynitride.paramdb import hbar

def check_POP_normalization(phi, q, w):
    r""" Checks the normalization of a POP potential.

    The normalization is given as

    .. math::

        \begin{align}
          \frac{\hbar}{2\omega} = \int dz \varepsilon_\infty(\omega^2_{LO}-\omega^2_{TO})\left(
              \left(\frac{\partial_z\phi}{\omega_{TO\parallel}^2-\omega^2}\right)^2 +
              \left(\frac{q\phi}{\omega^2_{TO\perp}-\omega^2}\right)^2 \right)
        \end{align}

    for the given POP potential `phi`.

    Args:
        phi: the hopefully normalized potential, should of course be provided on the full mesh that
            it's solved on, not just on some submesh where it interacts with carriers.
        q: the in-plane wavevector for the mode
        w: the angular frequency for the mode

    Returns:
        None

    Exceptions:
        AssertionError if the mode is not normalized
    """
    m = phi.mesh
    ew2 = m.eps_inf * ((m.wLO_para ** 2 - m.wTO_para ** 2) + (m.wLO_perp ** 2 - m.wTO_perp ** 2)) / 2
    parapart = (ew2 * ((phi.differentiate() / (m.wTO_para ** 2 - w ** 2)) ** 2)).integrate(definite=True)
    perppart = (ew2 * ((q * phi.tmf() / (m.wTO_perp ** 2 - w ** 2)) ** 2)).integrate(definite=True)
    norm = parapart + perppart
    req = hbar / (2 * w)
    assert np.isclose(norm, req, rtol=5e-2), "Mode normalization is off by a factor of {:.2g}".format(norm / req)


def check_POP_interface(phi, q, w):
    r""" Checks the continuity of the potential and displacement field for a POP phonon.

    Checks the continuity of :math:`\phi` and  :math:`\varepsilon_\parallel \frac{\partial phi}{\partial z}`
    at the interface.  These should be satisfied elsewhere automatically since the solutions are analytic.

    The absolute tolerance for :math:`D` discontinuity is set by the maximum value of :math:`D`.

    Args:
        phi: the hopefully normalized potential, should of course be provided on the full mesh that
            it's solved on, not just on some submesh where it interacts with carriers.
        q: the in-plane wavevector for the mode
        w: the angular frequency for the mode

    Returns:
        None

    Exceptions:
        AssertionError if the mode fails interface conditions

    """
    m = phi.mesh

    # Interface adjacent quantities
    iu, il = m.interfaces_mid[0][:2]
    eps_para_u = (m.eps_inf * (m.wLO_para ** 2 - w ** 2) / (m.wTO_para ** 2 - w ** 2))[iu]
    eps_para_l = (m.eps_inf * (m.wLO_para ** 2 - w ** 2) / (m.wTO_para ** 2 - w ** 2))[il]
    dphidz_u = phi.differentiate()[iu]
    dphidz_l = phi.differentiate()[il]

    # Scale for D
    eps_para = (m.eps_inf * (m.wLO_para ** 2 - w ** 2) / (m.wTO_para ** 2 - w ** 2))
    dphidz = phi.differentiate()
    D = dphidz * eps_para
    Dscale = np.max(np.abs(D))

    # If there's a spike in the derivative of phi at the interface,
    # that indicates discontinuity in phi
    assert np.isclose(dphidz_u, phi.differentiate()[iu - 1], rtol=.5, atol=Dscale / 1e2), \
        "phi is discontinuous at interface"
    assert np.isclose(dphidz_l, phi.differentiate()[il + 1], rtol=.5, atol=Dscale / 1e2), \
        "phi is discontinuous at interface"

    # The derivatives of phi on the sides of the interface
    # should be in ratio with the dielectric functions
    Du, Dl = eps_para_u * dphidz_u, eps_para_l * dphidz_l
    assert np.isclose(Du, Dl, rtol=1e-1, atol=Dscale / 1e2), \
        "D is not continuous at interface"

