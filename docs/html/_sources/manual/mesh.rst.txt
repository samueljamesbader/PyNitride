.. _mesh:

Meshing Scheme
=====================================
This page describes the properties of the mesh scheme implemented in :py:mod:`~pynitride.poissolve.mesh`.

Dual mesh
-----------------------------
The "main" meshes in PyNitride are referred to as *node meshes*.  But for every node mesh, there is a dual *mid mesh*
whose points are simply the midpoints between the node mesh locations.  In general solution variables and all other
continuous functions are defined on the node mesh.  There is guaranteed to be aligned to all interfaces; conversely,
the mid mesh will never contain a point at an interface; all points of the mid mesh are specifically in one layer.  This
makes the mid mesh the correct place to put material properties which are intrinsically discontinuous at interfaces.

Differentiating or integrating a function on one mesh "naturally" produces a function defined on the dual mesh, but any
function can be re-interpolated between the two using the :py:func:`~pynitride.mesh.Function.tpf` and
:py:func:`~pynitride.mesh.Function.tmf` functions.


Li_1994_ discusses empirically (in the context of effective mass discontinuities in the Schrodinger equation) the
preferred method to address the ambiguity of of the material property :math:`a` at an interface and the effect of that
choice on the convergence of the solution.
But in this formalism, there is no ambiguity, because :math:`a` is only ever defined and requested on the mid mesh.
Note that the :py:mod:`~pynitride.tests.Li1992_SQW.test_SQW_convergence` test shows that, empirically, this
discretization appears to work similarly to their recommendation.

Shared values on the mesh
-----------------------------
Different solvers, carrier models, materials, and such all interact through the variables shared on a
:py:mod:`~pynitride.mesh.Mesh`.  In writing any of these components, it is important to understand how the variables get
on the mesh and the rules for using them safely.


Materials
.............

.. _Li_1994: http://www.sciencedirect.com/science/article/pii/S0021999184710266

