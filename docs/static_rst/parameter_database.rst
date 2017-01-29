.. _parameter_database:

The global parameters database
=====================================

Where are the parameters?
-------------------------------------
Parameter sets and configurations available throughout PyNitride are stored as ``.txt`` files in the ``PyNitride/parameters`` directory.

Files that begin with an underscore (eg ``_meta.txt``) are for configuration and all others are (intended) for material parameters and the like.
The first line of a file describes its syntax.

- ``_system.txt`` begins with a line ``# Pint``, implying that it is a units definition file for the Pint_ library.
- Currently, all other files supplied in the distribution begin with ``PyNitride v2``, indicating compatibility with version 2 of PyNitride.

Accessing parameters
-------------------------------------
Parameters can be accessed from a :py:class:`~pynitride.paramdb.ParamDB` object.  They will be returned in the internal unit system of PyNitride.  For example, if the relative dielectric constant of GaN is :math:`\epsilon_r=10.4`, then ::

    >>> from pynitride import *
    >>> pdb=ParamDB()
    >>> epsilon=pdb['GaN.dielectric.eps']
    >>> print("{:.3g} eps_0".format(to_unit(epsilon,"epsilon_0")))
    10.4 eps_0
    >>> print("{:.2g} F/cm".format(to_unit(epsilon,"F/cm")))
    9.2e-13 F/cm

The above example shows how to print the permittivity of GaN in two ways. The details of the units system are discussed in the :ref:`Units <units>` section, so here we will just discuss the syntax for accessing the parameters.


[ FILL IN A LOT MORE HERE. ]

Parameters file format
-------------------------------------

Overall structure
.....................................


Parameter files to be used by PyNitride version 2 should all observe the following conventions

- The first line of the file is ``PyNitride v2``
- Any content after ``#`` on a line is ignored (as a comment)
- Blank lines (including lines which begin with ``#``) are ignored
- Except where specifically mentioned, the characters ``=``, ``.``, and ``:`` should not be used

There are three types of lines:

**Unnamed lines**, eg ``polarization``
    are simply a string ``Y``.  This indicates that the subsequent parameters belong to ``Y``.  For this example ``polarization`` might be followed by lines describing coefficients of spontaneous and piezoelectric polarization.
**Named lines**, eg ``material=GaN``
    are of the form ``X=Y`` where ``X`` and ``Y`` are any strings.  ``Y`` has the same role as above: in this example, lines would follow describing the material ``GaN``.  The ``X=`` is provided as a convenience which will allow more flexible forms of access as described in `Accessing parameters`_.  Also ``Y`` can be just a single period (``.``): this typically indicates a default value.  For example, there are often lines like ``conditions=relaxed`` which indicate parameters for a relaxed material, as well as lines like ``conditions=.`` to indicate fallback parameters which apply when conditions are not specified.
**Value lines**, eg ``eps: 10.4 epsilon_0``
    are of the form ``X:Y``, where ``X`` is any string (with no quotes, not containing periods or equals signs) and ``Y`` is a value as described below in `Values specification`_.  When following the below-specified formats ``Y`` can contain any of the mentioned special characters as needed.

Lines are related by nesting space-indentation::

    ...
    binary=GaN
        conditions=strained to AlN
            Eg: 3.605 eV
            surface=GenericMetal
                electronbarrier: 1 eV
            carrier=electron
                DEc: 0 eV
                band=.
                    g: 2
                    mdos: .2 m_e
            carrier=hole
                DEv: 0 eV
                band=HH
                    g: 2
                    mdos: 1.5 m_e
            polarization
                Ptot: 2e-6 e/cm**2    # don't trust this value
    ...
Note that lines of different types all intermix, but also that it is important to align indents (eg. ``Eg``, ``surface=...``, ``carrier=...`` are all the same number of spaces from the start of line.  This example provides a couple parameters GaN strained to AlN: a bandgap, a Schottky barrier, and degeneracies including spin for an electron band and a hole band.  The format of the files has been discussed, but the actual structure of what information should be nested where is left to the individual solvers.

Values specification
.....................................

Values for **Value lines** may be of any of following forms

- A number with or without units. For example, ``g: 2`` describes a unitless degeneracy. And ``mdos: .2 m_e`` describes the density-of-states mass in units of the electron mass.  Such expressions will be parsed by Pint_, and a full list of allowed values (including units such as ``meter``, prefixed units such as ``meV``, more complex units like ``cm**-2`` and constants such as ``hbar``) is provided in the `Pint docs <https://github.com/hgrecco/pint/blob/master/pint/default_en.txt>`_.  Numbers with units are all converted to pure numbers in the internal units system of PyNitride.
- A string enclosed by quotes, eg ``type: "Acceptor"``
- An arbitrary single-line Python expression enclosed in backticks, eg ```Eg: lambda T: 3.510 - (.909)*T**2/(T+830)```.  Currently, such expressions do not have access to other parameter values, but that is an obvious direction for future development.

.. _Pint: https://pint.readthedocs.io/



