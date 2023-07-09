.. _mesh:

Meshing Scheme
=====================================
This page describes the properties of the mesh scheme implemented in the :py:mod:`~pynitride.core.mesh` module
and how variables are defined on the mesh through :py:class:`~pynitride.core.mesh.Function`.

Dual mesh
-----------------------------
A spatial mesh in PyNitride actually denotes two lists of points: the *node* points and the *mid* points.
Node points are the "main" mesh defined by the structure, and every layer interface is aligned on a node point.
Many direct solution variables, such as the electric potential, are defined as functions over the node points.
Mid points, conversely, are defined as the points halfway between each node point.
Because interfaces/ discontinuities align to node points, every mid point is within a well-defined layer.
So variables that depend on material properties, or are intrinsically discontinuous at interfaces,
are defined as functions over the mid points.

A mesh is represented by :py:class:`~pynitride.core.mesh.Mesh`, and the functions are
instances of :py:class:`~pynitride.core.mesh.Function` whose `pos` variable indicates whether
it is a node function or mid function.  Functions defined on a mesh can be retrieved like instance variables
(eg `mesh.Ec` to get the conduction band edge).
`Function` subclasses Numpy arrays, so that array operations work as they normally would.

If there are `n` node points, node functions act like a Numpy array whose last dimension is of length `n`.
Mid functions act as Numpy arrays whose last dimension is length `n-1`.
`Function` includes methods :py:func:`~pynitride.mesh.Function.tnf` ("to node function") and
:py:func:`~pynitride.mesh.Function.tmf` ("to mid function") which interpolate between the two possibilities.
(Note: `tnf` will do nothing on a node-function, so these can be used liberally, eg if you want the conduction band edge as
an array of length `n`, you can use `mesh.Ec.tnf()` without having to remember what `Ec` is actually defined on).

Functions have convenience methods, eg for calculus.  Differentiating or integrating a function on one pointset
naturally produces a function defined on the dual pointset
(eg since potential is a node function, electric field becomes a mid-function).

Li_1994_ discusses empirically (in the context of effective mass discontinuities in the Schrodinger equation) how
different ways of discretizing a material property discontinuity at an interface can affect the convergence
of the solution.  In a traditional single-mesh scheme, there's some ambiguity about how the equations are discretized
at interfaces, but this dual mesh setup ensures that material properties are well-defined where they are stored
and any interpolations are explicit. Note that the :py:mod:`~pynitride.tests.Li1992_SQW.test_SQW_convergence` test shows that, empirically, this
discretization appears to work similarly to Li et al's recommendation.

Shared values on the mesh
-----------------------------
Different solvers, carrier models, materials, and such all interact through the variables shared on a
:py:mod:`~pynitride.mesh.Mesh`.  In writing any of these components, it is important to understand how the variables get
on the mesh and the rules for using them safely.


Materials
.............

.. _Li_1994: http://www.sciencedirect.com/science/article/pii/S0021999184710266

