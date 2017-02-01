.. _mesh:

Meshing Scheme
=====================================
This page describes broadly the mathematical properties of the mesh scheme implemented in :py:mod:`~pynitride.poissolve.mesh`.

Dual mesh
-------------------
Poissolve employs a rather unique finite-difference meshing scheme, in that there are actually two interdigitated meshes defined, the *point mesh* and the *mid mesh*.  The point mesh is the "main one", and the mid mesh is simply defined by the midpoints of all points in the point mesh.  This scheme can be probably (?) be reinterpreted in terms of one mesh which is twice as dense as the main mesh, but there are intuitive reasons to distinguish the two. Properties of this distinction include:

- The point mesh is guaranteed to be aligned to all interfaces; conversely, the mid mesh will never contain a point on an interface.  Thus material properties are "naturally" defined on the mid mesh (...and can be interpolated to the main mesh if necessary.)
- Integrals and central-difference derivatives of a function defined on one mesh "naturally" produce a function defined on the other mesh.

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

