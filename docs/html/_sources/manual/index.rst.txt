.. PyNitride documentation master file, created by
   sphinx-quickstart on Wed Jan 11 16:15:59 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====================================
The PyNitride Scientific Package
=====================================

Everything the Python-friendly III-Nitride engineer needs!

What is PyNitride?
==================

.. figure:: FrontImage.png
    :alt: Quantum Well HEMT
    :scale: 65%
    :align: right
    :figwidth: 35%

    PyNitride band diagram of a
    GaN-on-AlN HEMT

PyNitride is a 1D solver for band diagram analysis of epitaxial heterostructures, and is capable of arbitrarily mixed
self-consistent simulations of classical, Schrodinger, and Multi-band k.p properties, as well as phonon spectra.  It can
be run on a laptop for simple jobs, and parallelizes well onto many-core machines for computationally intense
applications such as the study of hole-phonon interaction.

This codebase is in a continual state of work, but is steadily being refined
with validation and unit tests and more complete documentation.
I hope others might find these tools useful, either for their own calculation or for learning purposes,
as I've written up a good deal of the physics that goes into these problems.

Contents
=========

Find out more about the :ref:`math_and_physics` or skip into the :doc:`API reference <../auto/modules>`.


.. toctree::
    :maxdepth: 1

        Math and Physics <math_and_physics>
        Software Structure <overall_flow>
        API Reference <../auto/modules>


Authors
=========
.. image:: Authors.jpeg
    :alt: Sam Bader and Martin Schubert
    :scale: 85%
    :align: right

`Sam Bader <http://sambader.net>`_ is an Applied Physics PhD student with
`Prof Jena <https://djena.engineering.cornell.edu/>`_ and `Prof Xing <http://grace.engineering.cornell.edu/>`_
at Cornell University, working on p-channel III-Nitride devices.

`Martin Schubert <https://www.linkedin.com/in/mfschubert/>`_, PhD, is a Founder and Technical Lead
at X, the Moonshot Factory.

If you find this useful let me know! And show some love by citing the project!

Acknowledgements
==================
These tools owe a great deal to many other wonderful projects from the Python community,
including `Scipy <https://www.scipy.org/>`_,
`Cython <http://cython.org/>`_,
`Pint <https://pint.readthedocs.io/>`_,
and `Anaconda <https://www.continuum.io/>`_.

Thanks also to `1DPoisson <http://www3.nd.edu/~gsnider/>`_, the fast and convenient standalone 1D Schrodinger Poisson
solver from Greg Snider which inspired this toolset.

And finally, Sam thanks his advisors
`Debdeep Jena <https://djena.engineering.cornell.edu/>`_ and `Grace Xing <http://grace.engineering.cornell.edu/>`_
for research support and ever-helpful discussions.

.. Indices and tables
    ==================
..
    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`

