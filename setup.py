from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}

setup(
    name='PyNitride',
    version='0.1.2',
    packages=['pynitride'],
    url='',
    license='',
    author='Samuel James Bader',
    author_email='samuel.james.bader@gmail.com',
    description='Python utilities for 1D band diagrams and simulation',
    ext_modules = cythonize([
        "pynitride/maths.pyx",
        "pynitride/cython/assemblers/assemble1x1.pyx",
        "pynitride/cython/assemblers/assemble2x2.pyx",
        "pynitride/cython/assemblers/assemble3x3.pyx",

        "pynitride/cython_loops.pyx",
        #"tests/util/ctest_cython_loops.pyx",
        ], **ext_options),
    requires=['numpy', 'matplotlib', 'scipy', 'pytest', 'cython', 'pint'],
    include_dirs=[numpy.get_include(),ROOT_DIR]
)
