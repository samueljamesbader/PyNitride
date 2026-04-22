.. _dopants:

Dopant Models
=====================================
This page describes the physics of the dopant models implemented in :py:mod:`~pynitride.physics.solvers`.

Simple dopants
----------------

Donors
*******

Donors can take on two charge states, ionized (+1) and neutral/filled (0),
depending on the the Fermi level position relative to the donor energy level :math:`E`
(positive :math:`E` indicating distance below the conduction band edge).  The donor states may have degeneracy 
:math:`g`.  The donor is specified in the material parameter database as follows:


.. code-block:: text

    material=GaN
        dopant=Si
            type: 'Donor'
            E: 29.7 meV
            g: 2

The grand canonical partition function for such a donor subsystem is

.. math::
    Z = \exp\left(\frac{E_F}{k_B T}\right) + g \exp\left(\frac{E_F-(E_C-E)}{k_B T}\right)

resulting in an expected charge of

.. math::
    \langle q \rangle = \frac{1}{1 + g \exp\left(\frac{E_F-(E_C-E)}{k_B T}\right)}

for each donor site.  The density of donor sites is specified in the mesh definition.

Acceptors
*********
Acceptors can take on two charge states, ionized (-1) and neutral/filled (0).
The parameters are specified similarly, but the energy level :math:`E` is measured upwards from the valence band edge.

The expected charge is given by

.. math::
    \langle q \rangle = \frac{-1}{1 + g \exp\left(\frac{E_V+E-E_F}{k_B T}\right)}

DX Centers
**********
Experimental support for certain DX centers is enabled.
DX centers can show `a variety of behaviors <http://dx.doi.org/10.1063/1.4948245>`_ with long lifetimes,
but as PyNitride is a steady-state solver, we will take a simplified approach.
These DX centers are taken to have three states: ionized donor (+1), singly-occupied (0) at energy :math:`E_0` below the conduction band, and doubly-occupied (+1) at energy :math:`2E_1` below the conduction band.

The grand canonical partition function is given by

.. math::
    Z = 1
    + g_0 \exp\left(\frac{E_F-(E_C-E_0)}{k_B T}\right)
    + g_1 \exp\left(\frac{2E_F-2(E_C-E_1)}{k_B T}\right)

leading to a charge of

.. math::
    \langle q \rangle = \frac{1 - g_1 \exp\left(\frac{2E_F-2(E_C-E_1)}{k_B T}\right)}
    {Z}

In the cases considered, the singly-occupied state is significantly higher in energy than the double-occupied state, and thus its contribution to :math:`Z` is dominated by either of the other terms depending on the position of the Fermi level.
Taking this simplification, and assuming :math:`g_1=1` (as per, e.g. a two-fold degenerate donor occupied by two electrons),

.. math::
    \langle q \rangle \approx -\tanh\left(\frac{E_F-(E_C-E_1)}{k_BT}\right)

PyNitride employs this form for now, thus only requiring a single energy specified: 

.. code-block:: text

    material=AlN
        dopant=OxygenDX
            type: 'DX'
            E: 150 meV
            g: 1