.. _sharedvalues:

Shared values on the mesh
=====================================
Different solvers, carrier models, materials, and such all interact through the variables shared on a
:py:mod:`~pynitride.mesh.Mesh`.  In writing any of these components, it is important to understand how the variables get
on the mesh and the rules for using them safely.


.. _sharedvalued_materials:

Materials
-------------------
