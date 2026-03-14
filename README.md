# PyNitride

## System requirements
For typical usage, PyNitride just requires any system with Python >=3.14,
and a few standard packages which are installed implicitly
by following the [Installation instructions](#Installation).


*The following note is for developers building demanding numerical codes on top of PyNitride,
not really relevant to typical band-diagram generating users*: On *nix systems, PyNitride can be used with the `multiprocessing` fork-based parallelization.
Many classes will, behind the scenes, store their large data in a global structure
to avoid pickling it for each new forked process the class is passed to.
`multiprocessing` does not support forking on Windows, so on Windows this parallelization is turned off
in the config.ini file
(though C extensions such as Numpy may still be internally parallelized).

## Installation
### From Source

Since this project is still in pre-release and contains a few random chunks of my PhD work,
this is the only way to install.  After some cleaning up, I may put it on Pypi for easier access!

For building from source, Microsoft Visual C++ 14.0 (part of [Visual Studio Build Tools for C++](https://visualstudio.microsoft.com/visual-cpp-build-tools/)) is required.  The no-thinking option is during install to select the "Desktop development with C++" workload, though if you want to save space and figure out more specifically which packages are needed, be my guest.

Download this project via GitHub

    cd DIRECTORY_WHERE_I_WANT_TO_INSTALL
    git clone https://github.com/samueljamesbader/PyNitride
    
If you want to have an isolated virtual environment[^1], might as well make it hereand activate it. (The folder `venv/` is already in `.gitignore`).

    cd PyNitride
    python -m venv venv
    
    # Windows:
    venv\Scripts\activate.bat
    # *nix
    source venv/bin/activate

Now do a local editable install:

    pip install -e .

[^1]: Note to less-experienced Python users:
this is optional but having a virtual environment for each project is nice practice
so that you don't  mess up one project's Python when installing packages for another project.
Much cleaner than installing all python packages into your global system python
which will get cluttered real quick!
If you want to know more, search "python venv".
The catch is to make sure you activate the venv whenever you want to use it from a new terminal,
eg before opening a Jupyter notebook!  Running from an IDE like PyCharm will often take care of this all for you.) 
