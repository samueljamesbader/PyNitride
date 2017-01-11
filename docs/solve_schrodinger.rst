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

For each row :math:`\sigma`, the left off-diagonal is given by

.. math::
    \frac{\hbar^2}{2m_{z,\sigma-.5} dz_{\sigma-.5} \sqrt{dz_{\sigma-1}dz_{\sigma}}}

the diagonal is given by

.. math::
    \frac{\hbar^2}{2dz_{\sigma}}\left(\frac{1}{m_{z,\sigma-.5}dz_{\sigma-.5}}+\frac{1}{m_{z,\sigma+.5}dz_{\sigma+.5}}\right)

and the right off-diagonal is given by

.. math::
    \frac{\hbar^2}{2m_{z,\sigma+.5} dz_{\sigma+.5} \sqrt{dz_{\sigma+1}dz_{\sigma}}}




.. _Tan_1990: http://dx.doi.org/10.1063/1.346245
