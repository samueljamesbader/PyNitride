.. _DC:

Dielectric Continuum Model
========================================

The Dielectric Continuum model [#]_, describes optical phonons in bulk materials or heterostructures.  As presented
below, by solving for the associated phonon potential, it will find only modes which produce a potential, ie modes which
are *purely* transverse will not be identified.  In wurtzite materials, "ordinary" modes are purely transverse, but the
"extraordinary" modes, even those known as transverse optical (TO), do have some non-vanishing longitudinal component and
will appear from the following math.

Equations of Motion
-----------------------------
If the system is described by a driven oscillator equation

.. math::
    \newcommand{\abs}[1]{\left|#1\right|}
    \begin{equation}
      \mu\partial_t^2u_i =-\mu\omega_{0i}^2u_i+e^* E_i
    \end{equation}


where :math:`\mu` and :math:`i` runs over :math:`\perp, \parallel`. Then :math:`P_i=\frac{e^*}{V_u}u_i`, giving

.. math::

    \begin{align}
      P_i =\varepsilon_0\chi_iE_i, \quad \chi_i=\frac{e^{*2}}{\varepsilon_0\mu V_u}\frac{1}{\omega_{0i}^2-\omega^2}
    \end{align}


Defining an :math:`\varepsilon_i` so that :math:`D_i=\varepsilon_i E_i=\varepsilon^\infty E_i +P=(\varepsilon^\infty +\varepsilon_0\chi_i) E_i` leads to

.. math::
    \begin{equation}
      \varepsilon_\perp=\varepsilon^\infty\frac{\omega_{LO\perp}^2-\omega^2}{\omega_{TO\perp}^2-\omega^2},
      \quad\quad
      \varepsilon_\parallel=\varepsilon^\infty\frac{\omega_{LO\parallel}^2-\omega^2}{\omega_{TO\parallel}^2-\omega^2}
    \end{equation}

where :math:`\omega_{TO,i}^2=\omega_{0i}^2` and :math:`\omega_{LO, i}^2=\omega_{0i}^2+\frac{e^{*2}}{\mu V_u\varepsilon^\infty}`
The phonons are found by solving

.. math::

    \begin{equation}
      -\nabla\cdot\overset\leftrightarrow{\epsilon}\nabla\phi=0
    \end{equation}


which can be framed as an eigenvalue problem in :math:`q` at fixed :math:`\omega`

.. math::
    \begin{equation}
      \partial_z\epsilon_\parallel\partial_z\phi=q^2\epsilon_\perp\phi
    \end{equation}


Normalization
------------------
The :math:`\phi` must then be normalized by the prescription :math:`\int dz\frac{\mu}{V_u}\abs{\vec u}^2=\frac{\hbar}{2\omega}`.  To do this, we'll need to extract :math:`\vec u` from the :math:`\phi`:

.. math::
    \begin{align}
      \vec u = \frac{V_u}{e^*}\vec P=\frac{V_u}{e^*}\varepsilon_0\chi\vec E=-\frac{V_u}{e^*}\chi\nabla\phi
    \end{align}
So

.. math::

    \begin{align}
      \vec u = -\frac{\varepsilon_0V_u}{e^*}\left( \chi_\parallel \partial_z\phi \hat{z} +iq\chi_\perp\phi\hat{q} \right)
    \end{align}

.. math::

    \begin{align}
      \frac{\mu}{V_u}\abs{\vec u}^2 = \frac{\varepsilon_0\mu V_u}{e^{*2}}\left( (\chi_\parallel \partial_z\phi)^2 +q^2(\chi_\perp\phi)^2 \right)
    \end{align}

.. math::

    \begin{align}
      \frac{\mu}{V_u}\abs{\vec u}^2 = \frac{e^{*2}}{\mu V_u}\left( \left(\frac{\partial_z\phi}{\omega_{TO\parallel}^2-\omega^2}\right)^2 +\left(\frac{q\phi}{\omega^2_{TO\perp}-\omega^2}\right)^2 \right)
    \end{align}

.. math::

    \begin{align}
      \frac{\mu}{V_u}\abs{\vec u}^2 = \varepsilon_\infty(\omega^2_{LO}-\omega^2_{TO})\left( \left(\frac{\partial_z\phi}{\omega_{TO\parallel}^2-\omega^2}\right)^2 +\left(\frac{q\phi}{\omega^2_{TO\perp}-\omega^2}\right)^2 \right)
    \end{align}

where, since :math:`\omega_{LO\perp}` et al are taken from experiment, we'll average the :math:`(\omega_{LO}^2-\omega_{TO}^2)` factor over :math:`\perp, \parallel`.
Thus the normalization condition is written in terms of the potential as

.. math::

    \begin{align}
      \frac{\hbar}{2\omega} = \int dz \varepsilon_\infty(\omega^2_{LO}-\omega^2_{TO})\left( \left(\frac{\partial_z\phi}{\omega_{TO\parallel}^2-\omega^2}\right)^2 +\left(\frac{q\phi}{\omega^2_{TO\perp}-\omega^2}\right)^2 \right)
    \end{align}

.. _BWH:

Binary Wurtzite Heterojunction
---------------------------------------

Following the quantum well example of Komirenko1999_, :class:`~pynitride.phonons.DielectricContinuum_SWH` solves a
single heterojunction structure with uniform :class:`~pynitride.material.Wurtzite` regions 1/2, assuming the bottom region is semi-infinite.
The top surface at :math:`z=0` is assumed Dirichelet.
The thickness of the top layer is :math:`t_1` and the normalization thickness for the bottom layer which will be used to
break the continuum of states by an artificial Dirichelet condition is :math:`t_2`.
It is required that the frequencies be in this order:

.. math::

    \omega_{TO\perp, s}<
    \omega_{TO\parallel, s}<
    \omega_{TO\perp, f}<
    \omega_{TO\parallel, f}<
    \omega_{LO\perp, s}<
    \omega_{LO\parallel, s}<
    \omega_{LO\perp, f}<
    \omega_{LO\parallel, f}

where `s`/`f` refer to the slower/faster material.  This is satisfied for GaN/AlN, and for GaN/AlGaN if the Aluminium
composition is sufficiently high (see Komirenko1999_).  The class checks this assertion so it is safe to experiment.

In a given region, solutions are oscillating if :math:`\varepsilon_\perp\varepsilon_\parallel<0`
and exponential if :math:`\varepsilon_\perp\varepsilon_\parallel>0`.
At an interface, the derivative switches signs iff :math:`\varepsilon_{1\parallel}\varepsilon_{2\parallel}<0`.
We  will use the following convenient definitions, similar to the notation of Komirenko1999_
but for a factor of two in :math:`\alpha`

.. math::

  \xi_{i}=\sqrt{\abs{\varepsilon_{i\perp}\varepsilon_{i\parallel}}},\quad
  \alpha_i=\sqrt{\abs{\varepsilon_{i\perp}/\varepsilon_{i\parallel}}}

Then the vertical wavevector of a mode in a given region is :math:`k_i=q\alpha_i`.

If the solution is written in Region 1 with a normalization constant :math:`A` and Region 2 with a normalization constant :math:`B`, then the first matching condition gives us :math:`A/B`, and the normalization condition will be written

.. math::

    \begin{equation}
      A^2=\frac{\hbar}{2\omega}\big/\left[ \beta^2_{\parallel 1}\gamma_{\parallel 1}^2+\beta^2_{\perp 1}\gamma_{\perp 1}^2 + \left(\frac{B}{A}\right)^2\left(\beta^2_{\parallel 2}\gamma_{\parallel 2}^2+\beta^2_{\perp 2}\gamma_{\perp 2}^2 \right) \right]
    \end{equation}
with

.. math::

    \begin{equation}
      \beta_{\parallel i}^2=\varepsilon^\infty_i\left( \omega_{LOi}^2-\omega_{TOi}^2 \right)\left( \frac{k_i}{\omega_{TO\parallel i}^2-\omega^2} \right)^2
    \end{equation}

.. math::

    \begin{equation}
      \beta_{\perp     i}^2=\varepsilon^\infty_i\left( \omega_{LOi}^2-\omega_{TOi}^2 \right)\left( \frac{  q}{\omega_{TO\perp i}^2-\omega^2} \right)^2
    \end{equation}
and

.. math::

    \begin{equation}
      \gamma_{\parallel i}^2=\int_{i} dz \left(\frac{\partial_z \phi}{A k_i}\right)^2, \quad
      \gamma_{\perp i}^2=\int_{i} dz \left(\frac{\phi}{B}\right)^2
    \end{equation}

Confined to Region 1
.........................

If the solution is oscillating in Region 1 and decaying in Region 2, we can write

.. math::

    \begin{align}
      \phi_1=A\sin(k_1 z), \quad \phi_2=Be^{-k_2z}
    \end{align}

Matching interface conditions gives

.. math::

    \begin{equation}
      q=\frac{1}{\alpha_1 t_1}\left[ \tan^{-1}\left( \xi_1/\xi_2 \right) +\pi n \right]
    \end{equation}

with :math:`B/A=\sin(k_1 t_1)e^{k_2t_1}` and

.. math::

    \begin{equation}
      \gamma_{\parallel 1}=\frac{1}{2}(t_1+\frac{1}{2k_1}\sin(2k_1t_1)), \quad
      \gamma_{\perp     1}=\frac{1}{2}(t_1-\frac{1}{2k_1}\sin(2k_1t_1))
    \end{equation}

.. math::

    \begin{equation}
      \gamma_{\parallel 2}=\frac{1}{2k_2}e^{-2k_2t_1}, \quad
      \gamma_{\perp     2}=\frac{1}{2k_2}e^{-2k_2t_1}
    \end{equation}
Note that if the energy region adjoins :math:`\omega_{LO\perp 1}`, the above does admit one peculiar :math:`n=0` mode which has a very small :math:`q` for all :math:`\omega`.  This mode thus does not contribute to in-plane scattering, but is terrible for the math, so I'd filter it out.


Confined to Interface
..........................

If the solution is decaying in both regions, we can write

.. math::

    \begin{align}
      \phi_1=A\sinh(k_1 z), \quad \phi_2=Be^{-k_2z}
    \end{align}

Matching interface conditions gives

.. math::

    \begin{equation}
      q=\frac{1}{2\alpha t_1}\log\left[ \frac{\xi_2+\xi_1}{\xi_2-\xi_1} \right]
    \end{equation}

with :math:`B/A=\sinh(k_1t_1)e^{k_2t_1}` and

.. math::

    \begin{equation}
      \gamma_{\parallel 1}=\frac{1}{2}\left(\frac{1}{2k_1}\sinh(2k_1t_1)+t_1\right), \quad
      \gamma_{\perp     1}=\frac{1}{2}\left(\frac{1}{2k_1}\sinh(2k_1t_1)-t_1\right)
    \end{equation}

.. math::

    \begin{equation}
      \gamma_{\parallel 2}=\frac{1}{2k_2}e^{-2k_2t_1}, \quad
      \gamma_{\perp     2}=\frac{1}{2k_2}e^{-2k_2t_1}
    \end{equation}

For large :math:`q` mode approaches an resonant energy of :math:`\omega_{IF}` defined by :math:`\xi_1=\xi_2`.

Confined to Region 2
........................

If the solution is decaying in Region 1 and oscillating in Region 2, we can write

.. math::

    \begin{align}
      \phi_1=A\sinh(k_1 z), \quad \phi_2=B\sin(k_2 z+\theta)
    \end{align}


Matching interface conditions gives

.. math::

    \begin{align}
      \theta=\tan^{-1}\left( \frac{\xi_2}{\xi_1}\tanh(k_1t_1) \right)-k_2t_1
    \end{align}
with :math:`B/A=\sinh(k_1t_1)/\sin(k_2t_1+\theta)`.  The :math:`t_2` thickness normalization gives :math:`k_2=\pi (n+1)/t_2`, so

.. math::

    \begin{equation}
      q=\frac{\pi(n+1)}{\alpha_2t_2}
    \end{equation}

Normalization is accounted for via

.. math::

    \begin{equation}
      \gamma_{\parallel 1}=\frac{1}{2}\left(\frac{1}{2k_1}\sinh(2k_1t_1)+t_1\right), \quad
      \gamma_{\perp     1}=\frac{1}{2}\left(\frac{1}{2k_1}\sinh(2k_1t_1)-t_1\right)
    \end{equation}

.. math::

    \begin{equation}
      \gamma_{\parallel 2}=\frac{t_2}{2}, \quad
      \gamma_{\perp     2}=\frac{t_2}{2}
    \end{equation}

.. [#] as discussed in Stroscio and Dutta, *Phonons in Nanostructures*

.. _Komirenko1999: http://dx.doi.org/10.1103/PhysRevB.59.5013
