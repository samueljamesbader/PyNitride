.. _FEM:

Finite Element Method
=====================================

The most general form of differential equation faced by PyNitride is

.. math::
    \begin{equation}
        \left[C^0(z)-iC^L(z)\partial_z-i\partial_zC^R(z)-\partial_z C^2(z) \partial_z \right] f(z) = w(z)b(z)
    \end{equation}

where the :math:`C` are potentially matrices, :math:`C^L` and :math:`C^R` are transposes of one another.
The :math:`C` and :math:`w` are material properties (ie they are well-defined on the mid mesh)
whereas :math:`f(z)` and :math:`b(z)` are solution variables (well-defined on the node mesh).
This can include :math:`b(z)=\lambda f(z)` as an eigenvalue problem.

Let :math:`\Delta_i(z)` be the linear interpolation of :math:`\delta_{z_i, z_j}` on dependent variable :math:`z_j`.
Choose the basis functions :math:`\Delta_i` where :math:`i`, where :math:`i` ranges over all the nodes except Dirichelet
boundaries.

Now for each basis element, multiply the differential equation by that element and integrate.  Treat the material
properties as being step discontinuous to preserve sharp heterointerfaces.
Wherever there is a material property to the right of a differential operator, use integration by parts to flip the
differential operator onto the basis element instead.  This gives us a :math:`N-d` equations in :math:`N-d` unknowns
where :math:`N` is the number of nodes and :math:`d` is the number of Dirichelet boundary nodes. [*]_

This equation can then be written :math:`Af=Mb` or, for an eigenvalue problem, :math:`Af=\lambda Mf`,
where :math:`A` is the "stiffness matrix" and :math:`M` is the "load matrix."  The stiffness matrix is given by

.. math::
    \begin{align}
      A_{i,i-1}=
        &\frac{C^0_{i-.5}dz_{i-.5}}{6}+\frac{i}{2}\left( C^L_{i-.5}+C^R_{i-.5} \right)-\frac{C^2_{i-.5}}{dz_{i-.5}}\\
      A_{i,i+1}=
        &\frac{C^0_{i+.5}dz_{i+.5}}{6}-\frac{i}{2}\left( C^L_{i+.5}+C^R_{i+.5} \right)-\frac{C^2_{i+.5}}{dz_{i+.5}}\\
      A_{i,i}=
        &\frac{C^0_{i-.5}dz_{i-.5}}{3}+\frac{C^0_{i+.5}dz_{i+.5}}{3}\\
        &+\frac{i}{2}\left( -C^L_{i-.5}+C^L_{i+.5} + C^R_{i-.5}-C^R_{i+.5}  \right)\\
        &+\frac{C^2_{i-.5}}{dz_{i-.5}}+\frac{C^2_{i+.5}}{dz_{i+.5}}
    \end{align}

And the load matrix can be written

.. math::
    \begin{align}
      M_{i,i-1}&= \frac{w_{i-.5}dz_{i-.5}}{6}\\
      M_{i,i+1}&= \frac{w_{i+.5}dz_{i+.5}}{6}\\
      M_{i,i}&= \frac{w_{i-.5}dz_{i-.5}}{3}+\frac{w_{i+.5}dz_{i+.5}}{3}
    \end{align}


.. [*] Note that, even at a Dirichelet boundary the next node in has support that extends all the way to the Dirichelet node,
    ie the edge of the solution domain, where it falls to zero.  Thus the material parameters in that mid region at the edge
    will come into play even though that region is outside the range of node points actually used in deriving the equations
    below.

    Contrast this to a Neumann boundary where the last node is at the edge of the solution domain, so there is no
    support of that basis element extending onto the outer side of the node.  In the case of a Neumann boundary, the
    equations below which apply to each :math:`i` have to be taken with the prescription that any term evaluated outside the
    node range is ignored.  In the case of a Dirichelet boundary, that prescription does not matter since no terms will ever
    be outside the solution domain, because the solution domain is one node further than the set of equations.


