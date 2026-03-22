.. _carriers:

Carrier Models
=====================================
This page describes the physics of the carrier models implemented in :py:mod:`~pynitride.carriers`.

.. _carriers_semiclassical:

Semiclassical
-------------------
The Semiclassical model is the simplest carrier occupation scheme, employing only the three-dimensional semiconductor bulk bands and Fermi occupation statistics

.. math::
    :nowrap:

    \begin{align}
      n(z)=\sum_lN^l_{C}(z)\mathcal{F}_{1/2}\left(\frac{E_F(z)-E_C(z)-\Delta E^l_{C}(z)}{kT}\right)\\
      p(z)=\sum_lN^l_{V}(z)\mathcal{F}_{1/2}\left(\frac{E_V(z)-\Delta E^l_{V}(z)-E_F(z)}{kT}\right)
    \end{align}

The :math:`l` is a sum over subbands and the :math:`\Delta E^l_{C/V}` are subband-specific offsets from the band-edge.
For instance, the split-off subband in GaN can be hundreds of meV away from the band edge under compressive strain.

The subband-specific conduction/valence band effective DOS are given by

.. math::
    N^l_{C/V}(z)=g_vg_s\left(\frac{2m^l_\mathrm{3D}(z)kT}{2\pi\hbar^2}\right)^{3/2}

where  :math:`m^l_{3D}` is the geometric mean of three directional effective masses
and :math:`\mathcal{F}_{j}(\eta)` is the Fermi-Dirac integral of order :math:`j`:

.. math::
  \mathcal{F}_{j}(\eta)=\frac{1}{\Gamma(j+1)}\int_0^\infty\frac{x^jdx}{1+e^{x-\eta}}

The derivatives with respect to the bands are given by

.. math::
    :nowrap:

    \begin{align}
      n'(z)=-\sum_l\frac{N^l_{C}(z)}{kT}\mathcal{F}_{-1/2}\left(\frac{E_F(z)-E_C(z)-\Delta E^l_{C}(z)}{kT}\right)\\
      p'(z)=+\sum_l\frac{N^l_{V}(z)}{kT}\mathcal{F}_{-1/2}\left(\frac{E_V(z)-\Delta E^l_{V}(z)-E_F(z)}{kT}\right)
    \end{align}


Schrodinger
------------------
The Schrodinger model solves for the eigenstates of each subband separately.

Full k-space
................
The "full k-space" Schrodinger option [not implemented yet]
solves the Schrodinger problem with the BenDaniel and Duke Hamiltonian at a range of points in k-space and integrates up the density.
For the conduction and valence bands respectively

.. math::
    :nowrap:

    \begin{align}
      \left[-\partial_z\frac{\hbar^2}{2m^l_z(z)}\partial_z +\frac{\hbar^2k_t^2}{2m^l_t(z)}+ E_C(z)+\Delta E_C^i(z) - E_{i, k_t}\right]\psi_{i, k_t}(z)=0\\
      \left[\partial_z\frac{\hbar^2}{2m^l_{z}(z)}\partial_z-\frac{\hbar^2k_t^2}{2m^l_t(z)} + E_V^l(z) -\Delta E_V^i(z) - E^l_{i, k_t}\right]\psi^l_{i, k_t}(z)=0
    \end{align}

where the :math:`l` is a subband index, :math:`m_z` and :math:`m_t` are the longitudinal and transverse effective masses,
and the :math:`\Delta E_{C/V}` are subband-specific offsets.  These levels are occupied as

.. math::
    :nowrap:

    \begin{align}
      n(z)=g_sg_v\sum_{i, l}\int_{k_t}2\pi k_tdk_t \left|\psi^l_i\right|^2 /\left(1+\exp\left\{\frac{E_{i, k_t}-E_F}{kT}\right\}\right)\\
      p(z)=g_sg_v\sum_{i, l}\int_{k_t}2\pi k_tdk_t \left|\psi^l_i\right|^2 /\left(1+\exp\left\{\frac{E_F-E^l_{i, k_t}}{kT}\right\}\right)
    \end{align}

Note that, even though a parabolic E-k relation is assumed for the bulk carrier, the resulting energy levels may not be parabolic
with respect to transverse momentum because the transverse kinetic term is spatially varying in a general heterostructure, complicating
the dispersion of the energy levels.  However, often in relevant problems, the carrier is confined to mainly one material and the non-uniformity
of the transverse effective mass is merely a perturbation.  In such cases, the parabolic k-space solution may suffice.

Parabolic k-space
....................
The "parabolic" Schrodinger option solves the Schrodinger problem at :math:`k_t=0` and then, for each wavefunction, averages harmonically
over the transverse effective masses which that wavefunction sees to compute the appropriate effective mass to use for the parabolic transverse
dispersion of that level.

The Schrodinger equation is the same as above (only solved at at :math:`k_t=0`), but the carrier density expressions are simplified:

.. math::
    :nowrap:

    \begin{align}
      \left\{n/p\right\}(z)=g_sg_v\sum_{i,l}\left|\psi^l_i\right|^2 \frac{\overline{m}^l_{t,i}kT}{2\pi\hbar^2}\ln\left[1+\exp\left\{\eta^l_{i, n/p}\right\}\right]\\
    \end{align}

with

.. math::
    :nowrap:

    \begin{equation}
    \eta^l_{i, n}=\frac{E_F-E^l_i}{kT},\quad \eta^l_{i, p}=\frac{E^l_i-E_F}{kT}
    \end{equation}

and

.. math::
    :nowrap:

    \begin{equation}
      1/\overline{m}^l_{t,i}=\int dz \frac{\left|\psi_i^l(z)\right|^2}{m^l_{t}(z)}
    \end{equation}

and the derivatives

.. math::
    :nowrap:

    \begin{align}
      \left\{n'/p'\right\}(z)=g_sg_vb\sum_{i,l}\left|\psi^l_i\right|^2 \frac{\overline{m}^l_{t,i}}{2\pi\hbar^2}\frac{\exp\left\{\eta^l_{i, n/p}\right\}}{1+\exp\left\{\eta^l_{i, n/p}\right\}}\\
    \end{align}

with :math:`b=\{-1,+1\}` for :math:`\{n,p\}` respectively.

Multiband k.p
------------------
The Multiband k.p model solves for the energy eigenstates by generalizing the bulk bandstructure Hamiltonian from a
function of wavevector to a function of spatial derivatives.  In general, this will form a matrix differential equation

.. math::
    :nowrap:

    \begin{multline}
        \left[C^0(z,\vec k_\perp)-iC^L(z,\vec k_\perp)\partial_z
        -i\partial_zC^R(z,\vec k_\perp)-\partial_z C^2(z,\vec k_\perp) \partial_z \right] f(z,\vec k_\perp)\\
        = \lambda_n(\vec k_\perp) f(z,\vec k_\perp)
    \end{multline}

where the `C` matrices depend on the material system, z-dependent material properties, and in-plane wavevector.
The `C` matrices are returned by that material system
`kp_Cmats` function, eg the Wurtzite :func:`~pynitride.physics.material.Wurtzite.kp_Cmats`.

