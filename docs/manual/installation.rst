============
Installation
============

You can either install from PyPI (the easiest way) or build from source (if you want to contribute back changes or just want to run the latest code).

Installing from PyPI
=====================

If you just want to use PyNitride from your own Python code,
the easiest way to install is via pip from `TestPyPI <https://test.pypi.org/project/pynitride/>`_
(it will move to PyPI proper in the future).

Set up and activate your Python 3.14 virtual environment, then run:

.. code-block:: bash

   pip install --index-url https://test.pypi.org/simple/ pynitride --extra-index-url https://pypi.org/simple --pre

If all goes well, you should be able to run an example such to verify the installation:

.. code-block:: bash

   python -m pynitride.examples.AlGaN_GaN_HEMT.hemt_example

Note to Windows users, if you get a message like
`ImportError: DLL load failed while importing fem: An Application Control policy has blocked this file.`,
then your computer is trying to protect you from running the compiled code since it was downloaded from the internet.
The easiest fix is to use `cmd` instead of PowerShell to run the above command.
After running it once, Windows will recognize the wheel as safe and you should
be able to run it from PowerShell in the future if that's your preference.
(Alternative approaches include "Set-ExecutionPolicy" or "Unblock-File" in PowerShell.)


.. _building-from-source:

Building from source
=====================

Prerequisites
*************

- You'll need `Python 3.14 <https://www.python.org/>`_.
- For building from source on Windows, Microsoft Visual C++ 14.0 (part of `Visual Studio Build Tools for C++ <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_) is required.
  The simple option is to select the "Desktop development with C++" workload, though if you want to save space and figure out specifically which packages are needed, be my guest.

Setup
*****

Clone the repository (or fork first and clone that if you want to contribute back changes!):

.. code-block:: bash

   git clone https://github.com/samueljamesbader/PyNitride.git

Next set up your Python environment.  (This may differ on your system.)

e.g. Linux/Mac with a venv:

.. code-block:: bash

   cd PyNitride
   python -m venv venv
   source venv/bin/activate

or Windows with a venv in PowerShell:

.. code-block:: powershell

   cd PyNitride
   python -m venv venv
   .\venv\Scripts\activate

Then install the locked development dependencies:

.. code-block:: bash

   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e ".[dev]" --no-deps
   pip check
