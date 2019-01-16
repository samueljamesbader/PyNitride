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
PyNitride began as a fun side-project, building a 1D Schrodinger-Poisson solver, and evolved into a cohesive, integrated collection of all of the Python code that I have found useful in my thesis work on Gallium Nitride Quantum Well High Electron Mobility Transistors.  This codebase is in a continual state of work, but is steadily being refined with validation and unit tests and more complete documentation.  I hope others might find these tools useful, either for their own calculation or for learning purposes, as I've written up a good deal of the physics that goes into these problems.

Contents
=========

Find out more about the :ref:`math_and_physics` of this solver or jump right into the :doc:`API reference <../auto/modules>`.


.. toctree::
    :maxdepth: 1

        Math and Physics <math_and_physics>
        Software Structure <overall_flow>
        Parameter Database <parameter_database>
        Units system <units>
        API Reference <../auto/modules>
        Source code on Github <https://github.com/samueljamesbader/PyNitride>


Author
=========
.. image:: SamBader.jpg
    :alt: Sam Bader
    :scale: 45%
    :align: right

If you have questions, feel free to connect with the project author (`Sam Bader <http://sambader.net>`_).

I'm an Applied Physics PhD student with the `Prof Jena <https://djena.engineering.cornell.edu/>`_ and `Prof Xing <http://grace.engineering.cornell.edu/>`_ at Cornell University, and I love talking about computational methods and device physics!

If you find this useful let me know! And show some love by citing the project!

Acknowledgements
==================
These tools owe a great deal to many other wonderful projects from the Python community, including `Scipy <https://www.scipy.org/>`_, `Cython <http://cython.org/>`_, `Pint <https://pint.readthedocs.io/>`_, `Anaconda <https://www.continuum.io/>`_, `VPython <http://vpython.org/>`_, and `pytest <http://doc.pytest.org/en/latest/>`_.

Thanks also to `1DPoisson <http://www3.nd.edu/~gsnider/>`_, the incredibly fast and convenient standalone 1D Schrodinger Poisson solver from Greg Snider which inspired this toolset.

And finally thanks to the encouragement of my advisors `Debdeep Jena <https://djena.engineering.cornell.edu/>`_ and `Grace Xing <http://grace.engineering.cornell.edu/>`_ as I polished this code when I should have been doing real work.

.. Indices and tables
    ==================
..
    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`

