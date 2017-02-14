.. _dataman:

Organizing and importing data
=====================================

The :py:class:`Omniscient subpackage <pynitride.omniscient>` provides utilities for reading in and manipulating the overflowing directories of data files commonly encountered in experimental work.

Expected organization
----------------------------
There is one central folder of data locally which Omniscient analyzes, by default specified in the PyNitride ``config.ini``.  If you wish to keep different datasets in different locations on the same system, you can certainly just populate this central folder with symlinks to other folders (no recursion please!).  Alternatively, when initializing a :py:class:`~pynitride.omniscient.dataman.DataManager`, you can provide a separate directory as an argument.  Whatever folder you use is hereafter referred to as ``DATA``.

Within ``DATA``, there can be any arbitrarily deep nesting of subdirectories, and you can import the contents of any of the
directories by specifying its path within ``DATA`` to the :py:func:`~pynitride.omniscient.dataman.DataManager.load_subproject`.  Data files will only be read if there is a file named ``_meta.key`` in their precise directory which indicates their metadata.  The bulk of the data organization "provided by" Omniscient is simply a matter of you specifying in the ``_meta.key`` file how the various data are treated.

The ``_meta.key`` format
-----------------------------

Basic options
______________

The simplest ``_meta.key`` file looks something like this::

    ### Description
    # Austin and I collected this data from ...

    ### File descriptors
    [_headers]
    power: float
    itime: float

    ### File schema
    [SingleSpectrum]
    nameregex: SingleSpec_(?P<power>\d\.+)uW_(?P<itime>\d\.+)s.txt
    reader: WITec_SingleSpec_TXT

The file is read by a standard Python :py:class:`~configparser.ConfigParser`, so it conforms to the config file format.  Namely, lines beginning with ``#`` are ignored.  Lines of the form ``[SECTIONNAME]`` delineate different sections, and key-value pairs can be given as ``KEY: VALUE`` (or many other ways).  Beyond that:

    - There must be one section ``[_headers]`` which contains a list of the metadata which will be tabulated for each file in this directory.
        - In this example, the only parameters which will be tabulated are a power and an integration time for each file.  For each parameter, a data type (``str``, ``int``, ``float``) is specified.

    - Other sections (which don't begin with an underscore) provide the schema for different file types.  The above example defines a measurement type and names it ``SingleSpectrum``.  The schema includes

        - a regular expression with `named groups <https://docs.python.org/3/library/re.html#regular-expression-syntax>`_ which Omniscient will use to recognize files fitting this schema.  The names should match up to the parameters in ``_headers``.
        - the name of the module in :py:mod:`~pynitride.omniscient.readers` which provides a ``read()`` function for parsing this file.  (If this option is not specified, Omniscient will check for a module named the same as the measurement type (ie ``Spectrum`` in this example.)

Given an example like the above, the Omniscient :py:class:`~pynitride.omniscient.DataManager` will check the directory of this ``_meta.key`` for files matching the regex, and produce a table which includes the filenames, the power and integration time used for each file (as extracted from the filename), and a data column which contains the result of ``pynitride.omniscient.WiTec_SingleSpec_TXT.read()`` called on that file.

More flexible options
____________________

This section will bring in some powerful features of Omniscient to make our data system more flexible::

    ### Description
    # Austin and I collected this data from ...

    ### File descriptors
    [_headers]
    power: float
    itime: float
    laser: str

    ### File schema
    # For spectra taken at one point
    [SingleSpectrum]
    nameregex: SingleSpec_(?P<power>\d\.+)uW_(?P<itime>\d\.+)s\.txt
    reader: WITec_SingleSpec_TXT

    # For spectra scanned across the substrate
    [AreaScan]
    nameregex: AreaSpec_(\d\.+)uW\.txt
    reader: WITec_AreaScan_TXT
    additionalinfo: {'itime':1.0,'laser':'488nm'}

    ### Futher log info
    # For the lower power SingleSpectrum measurements, we used 633nm laser
    # for all else, we used 488nm laser
    [_switchinglasers]
    if: (mtype=="SingleSpectrum")
    then: {'laser': '633nm' if power < 1 else '488nm'}

First, we've added another measurement type ``AreaScan``, which is read by a different reader module, and has a different file naming scheme.  For instance, the filename of the ``AreaScan`` measurements don't include the integration time.  But there is now a field ``additionalinfo`` which specifies that the integration time is 1s for all the ``AreaScan`` files.

Second, we've added one more parameter to track: a string indicating which laser was used.  This parameter is not explicitly given in the filenames (ie you don't see it in the regexes).  There is an ``additionalinfo`` field in the ``AreaScan`` type that says all the area scans were performed with the 488nm laser.

Third, there is a section with more log info.  Any section besides ``_headers`` which begins with an underscore is considered further log info.  These sections have a conditional specified by ``if:``, and a dictionary specified by ``then:``, which allows parameters to be supplied in arbitrarily complicated ways to sets of files en masse.  In this example, the log supplies the ``laser`` information for all of the ``SingleSpectrum`` scans.  Since we changed laser mid-course, and never updated the file name schema, it was easier to just "log" this information.  As seen in the example, ``if:`` and ``then:`` are Python expressions evaluated in such a context that they have access to the other already-defined variables for each file.

With these techniques, it is straightforward to organize data in a way that is easy for :py:class:`~pynitride.omniscient.dataman.DataManager` to tabulate over sets of files.

[EXAMPLE READIN]
