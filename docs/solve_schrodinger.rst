.. _solve_schrodinger:

=====================================
Solving the Schrodinger Equation
=====================================


This document describes the math of the :py:class:`~poissolve.solvers.schrodinger.SchrodingerSolver`.

.. _solve_schrodinger__1d:

---------------------------------------
Basic 1-D problem on a non-uniform mesh
---------------------------------------

Read Tan_1990_

The kinetic term is tri-diagonal and symmetric.  For each row :math:`\sigma`, the center term is

.. math::
    T_{\sigma,\sigma}=\frac{\hbar^2}{2dz_{\sigma}}\left(\frac{1}{m_{z,\sigma-.5}dz_{\sigma-.5}}+\frac{1}{m_{z,\sigma+.5}dz_{\sigma+.5}}\right)

And the left and right off-diagonal terms are respectively

.. math::
    T_{\sigma, \sigma-1}=\frac{\hbar^2}{2m_{z,\sigma-.5} dz_{\sigma-.5} \sqrt{dz_{\sigma-1}dz_{\sigma}}},\quad
    T_{\sigma, \sigma+1}=\frac{\hbar^2}{2m_{z,\sigma+.5} dz_{\sigma+.5} \sqrt{dz_{\sigma+1}dz_{\sigma}}}


.. _Tan_1990: http://dx.doi.org/10.1063/1.346245
