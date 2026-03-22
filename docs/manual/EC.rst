.. _EC:

Elastic Continuum Model
========================================

The elastic continuum model [#]_ joins the continuum Newton's law with the material stress-strain relation:

.. math::
    :nowrap:

    \begin{equation}
      \rho \frac{\partial^2 u_i}{\partial t^2}=\frac{\partial T_{ij}}{\partial r_j}, \quad T_{ij}=c_{ijkl}\epsilon_{kl}
    \end{equation}

where :math:`u_i` is the local displacement, :math:`\rho` is the density,
:math:`T_{ij}` is the stress tensor,
:math:`c_{ijkl}` is the stiffness tensor,
and :math:`\epsilon_{ijkl}` is the strain tensor
:math:`\epsilon_{ij}=\frac{1}{2}\left( \partial_{r_j} u_i + \partial_{r_i} u_j\right)`.
In Voigt notation

.. math::
    :nowrap:

    \begin{equation}
      T_{\alpha}=c_{\alpha \beta}\epsilon_\beta
    \end{equation}

where :math:`\alpha, \beta` run 1-6 and the Voigt tuples are related to the actual tensors by

.. math::
    :nowrap:

    \begin{align}
      T_{1}=T_{xx}&,\quad
      \epsilon_{1}=\epsilon_{xx}\\
      T_{2}=T_{yy}&,\quad
      \epsilon_{2}=\epsilon_{yy}\\
      T_{3}=T_{zz}&,\quad
      \epsilon_{3}=\epsilon_{zz}\\
      T_{4}=T_{yz}&,\quad
      \epsilon_{4}=2\epsilon_{yz}\\
      T_{5}=T_{xz}&,\quad
      \epsilon_{5}=2\epsilon_{xz}\\
      T_{6}=T_{xy}&,\quad
      \epsilon_{6}=2\epsilon_{xy}
    \end{align}

Wurtzite
------------

For a wurtzite crystal, the :math:`c_{\alpha \beta}` can be written

.. math::
    :nowrap:
    
    \begin{equation}
      c=\begin{pmatrix}
        C_{11} & C_{12} & C_{13} &      0 &      0 & 0 \\
        C_{12} & C_{22} & C_{13} &      0 &      0 & 0 \\
        C_{13} & C_{13} & C_{33} &      0 &      0 & 0 \\
        0      &      0 &      0 & C_{44} &      0 & 0 \\
        0      &      0 &      0 &      0 & C_{44} & 0 \\
        0      &      0 &      0 &      0 &      0 & \frac{1}{2}\left(C_{11}-C_{12}  \right)  \end{pmatrix}
    \end{equation}

with in-plane wavevector :math:`q`, the strains are given by

.. math::
    :nowrap:

    \begin{align}
      \epsilon_1&=iq u_x ,& \quad \epsilon_4&=\partial_z u_y \\
      \epsilon_2&=0 ,& \quad \epsilon_5&=iqu_z+\partial_z u_x \\
      \epsilon_3&=\partial_zu_z ,& \quad \epsilon_6&=iqu_y
    \end{align}

Then the combined relation becomes

.. math::
    :nowrap:

    \begin{equation}
      -\rho\omega^2\begin{pmatrix}
        u_x\\
        u_y\\
        u_z
      \end{pmatrix}=
      \begin{pmatrix}
        -C_{11}q^2u_x + iqC_{13}\partial_zu_z + \partial_zC_{44}iq u_z+\partial_zC_{44}\partial_zu_x  \\
        -\frac{1}{2}\left( C_{11}-C_{12} \right)q^2u_y +\partial_z C_{44}\partial_z u_y\\
      -C_{44}q^2u_z+C_{44}iq\partial_z u_x+iq\partial_zC_{13}u_x+\partial_zC_{33}\partial_zu_z
      \end{pmatrix}
    \end{equation}

Or

.. math::
    :nowrap:

    \begin{equation}
      \rho\omega^2u=
      Cu, \quad C= q^2C^0 -iq C^L\partial_z -iq \partial_z C^R  - \partial_z C^2 \partial_z
      \label{eq:split}
    \end{equation}

where

.. math::
    :nowrap:

    \begin{equation}
      C^0= \begin{pmatrix}
        C_{11} &  & \\
        & \frac{1}{2}\left( C_{11}-C_{12} \right) & \\
        & & C_{44}
      \end{pmatrix}, \quad
      C^2= \begin{pmatrix}
        C_{44} &  & \\
        & C_{44} & \\
        & & C_{33}
      \end{pmatrix}
    \end{equation}

.. math::
    :nowrap:

    \begin{equation}
      C^L= \begin{pmatrix}
        &  & C_{13}\\
        & 0& \\
        C_{44} & &
      \end{pmatrix}, \quad
      C^R= \begin{pmatrix}
        &  & C_{44}\\
        & 0& \\
        C_{13} & &
      \end{pmatrix}
    \end{equation}

Wurtzite piezoelectric potential
-------------------------------------

Once the acoustic phonon modes are solved for, each mode can be considered a source of piezoelectric charge

.. math::
    :nowrap:

    \begin{align}
    \rho=-\nabla \cdot \vec P&= -\nabla \cdot \left[
    \begin{pmatrix}
      0 & 0 & 0 & 0 & e_{15} & 0\\
      0 & 0 & 0 &  e_{15} & 0 & 0\\
      e_{31} & e_{31} & e_{33} & 0 & 0 & 0
    \end{pmatrix}
    \begin{pmatrix}
      \epsilon_{xx}\\
      \epsilon_{yy}\\
      \epsilon_{zz}\\
      2\epsilon_{yz}\\
      2\epsilon_{xz}\\
      2\epsilon_{xy}
    \end{pmatrix} \right]
    \end{align}

where :math:`e_{\alpha\beta}` are the piezoelastic moduli

.. math::
    :nowrap:

    \begin{equation}
      q^2\varepsilon_\perp\phi -\partial_z\varepsilon_\parallel \partial_z \phi
        =q^2e_{15}u_z-iqe_{15}\partial_zu_L
        -iq \partial_z e_{31}u_L- \partial_ze_{33}\partial_zu_z
    \end{equation}

which can be written

.. math::
    :nowrap:

    \begin{equation}
      C^0\phi -\partial_z C^2\partial_z\phi = C^{0'}u_z-iC^{L'}u_x-iC^{R'}u_x-\partial_zC^{2'}\partial_zu_z
    \end{equation}

where

.. math::
    :nowrap:

    \begin{equation}
      C_0=q^2\varepsilon_\perp, \quad C_2=\varepsilon_\parallel
    \end{equation}

.. math::
    :nowrap:

    \begin{equation}
      C^{0'}=q^2e_{15},
      C^{L'}=qe_{15},
      C^{R'}=qe_{31},
      C^{2'}=e_{33}
    \end{equation}

..  Note, this disagrees with Pokatilov by a factor of two in every term that includes an $e_{15}$.
    I think they, like Stroscio, forgot the factors of times-two engineering strains above.
    Those factors should be there according to Yang (https://doi.org/10.1007/978-3-030-03137-4)
    and Nye (Physical Properties of Crystals: Their Representation by Tensors and Matrices).


.. [#] as discussed in Stroscio and Dutta, *Phonons in Nanostructures*
