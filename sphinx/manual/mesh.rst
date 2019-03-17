.. _mesh:

Meshing Scheme
=====================================
This page describes the mathematical properties of the mesh scheme implemented in :py:mod:`~pynitride.poissolve.mesh`.

Dual mesh
-------------------
The "main" meshes in PyNitride are referred to as *point meshes*.  But for every point mesh, there is a dual *mid mesh*
whose points are simply the midpoints between the point mesh locations.  In general solution variables and all other
continuous functions are defined on the main mesh.  There is guaranteed to be aligned to all interfaces; conversely,
the mid mesh will never contain a point at an interface; all points of the mid mesh are specifically in one layer.  This
makes the mid mesh the correct place to put material properties which are intrinsically discontinuous at interfaces.

Differentiating or integrating a function on one mesh "naturally" produces a function defined on the dual mesh, but any
function can be re-interpolated between the two using the :py:func:`~pynitride.mesh.Function.tpf` and
:py:func:`~pynitride.mesh.Function.tmf` functions.


Example: generic differential equation
-----------------------------------------
For example, many equations we solve are of the form

.. math::
    \partial_z \left[a(z) \partial_z f(z) \right]= b(z)

where :math:`f` and :math:`b` are functions (such as electric potential and charge density), and :math:`a` is a material property (such as dielectric constant).  Then :math:`f` and :math:`b` are defined on the point mesh, and :math:`a` and :math:`\partial_z f` are defined on the mid mesh, and :math:`\partial_z [a \partial_z f]` is defined on the point mesh.

Employing a central-difference discrete derivative on a potentially non-uniform mesh, this produces an equation at each point :math:`\sigma`:

.. math::
    \frac{1}{dz^m_{\sigma}}\left[a_{\sigma+.5}\frac{ f_{\sigma+1}-f_\sigma}{dz^p_{\sigma+.5}}
    -a_{\sigma-.5}\frac{f_\sigma-f_{\sigma-1}}{dz^p_{\sigma-.5}}\right]=b_\sigma

where subscript :math:`\sigma+.5` is understood to indicate the midpoint between :math:`\sigma` and :math:`\sigma+1`, and :math:`dz^p` are the differences between subsequent point mesh positions (defined on the mid mesh) and :math:`dz^m` are the differences between subsequent mid mesh positions (defined on the point mesh).

Considering this expression where :math:`\sigma` happens to be an interface point reveals a beauty of this discretization scheme.  Li_1994_ discusses empirically (in the context of effective mass discontinuities in the Schrodinger equation) the preferred method to address the ambiguity of of the material property :math:`a` at an interface.  But in this formalism, there is no ambiguity, because :math:`a` is only ever defined and requested on the mid mesh.  Note that the :py:func:`~pynitride.poissolve.test_solvers.test_schrodinger_li1994` test shows that, empirically, this discretization appears to work similarly to their recommendation.


.. _Li_1994: http://www.sciencedirect.com/science/article/pii/S0021999184710266

