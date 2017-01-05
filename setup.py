from distutils.core import setup
from Cython.Build import cythonize

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}

setup(
    name='Poissolve',
    version='0.1.2',
    packages=['poissolve', 'poissolve.maths', 'poissolve.tests', 'poissolve.solvers'],
    url='',
    license='',
    author='Samuel James Bader',
    author_email='samuel.james.bader@gmail.com',
    description='Python utilities for 1D band diagrams and simulation',
    ext_modules = cythonize(["poissolve/maths/fermi_dirac_integral.pyx","poissolve/maths/tdma.pyx"], **ext_options),
    requires=['numpy', 'matplotlib', 'scipy', 'pytest', 'cython']
)
