from distutils.core import setup
from Cython.Build import cythonize
import numpy

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}

setup(
    name='Poissolve',
    version='0.1.2',
    packages=['poissolve', 'poissolve.maths', 'poissolve.tests', 'poissolve.solvers_old'],
    url='',
    license='',
    author='Samuel James Bader',
    author_email='samuel.james.bader@gmail.com',
    description='Python utilities for 1D band diagrams and simulation',
    ext_modules = cythonize(["pynitride/poissolve/maths.pyx","pynitride/compact/models.pyx"], **ext_options),
    requires=['numpy', 'matplotlib', 'scipy', 'pytest', 'cython', 'pint'],
    include_dirs=[numpy.get_include()]
)
