.. _dataman:

Reading in data
=====================================

The :py:class:`Omniscient subpackage <pynitride.omniscient>` provides utilities for reading in and manipulating the sorts of text-based data files commonly encountered in this field.

Expected organization
----------------------------
There is one central folder of data locally which Omniscient analyzes, by default specified in the PyNitride ``config.ini``.  If you wish to keep different datasets in different locations on the same system, you can certainly just populate this central folder with symlinks to other folders [1]_.  Alternatively, when initializing a :py:class:`~pynitride.omniscient.dataman.DataManager`, you can provide a separate directory as an argument.  Whatever folder you use is hereafter referred to as ``DATA``.

Within ``DATA``, there can be any arbitrarily deep nesting of subdirectories, and you can import the contents of any of the
directories by specifying its path within ``DATA`` to the :py:func:`~pynitride.omniscient.dataman.DataManager.load_subproject`.  Data files will only be read if there is a file named ``_meta.key`` in their direct parent directory which indicates their metadata.  The bulk of the data organization "provided by" Omniscient is simply a matter of you specifying in the ``_meta.key`` file how the various data are treated.

The ``_meta.key`` format
-----------------------------


.. _[1] but please don't create recursive loops of nesting directories!
