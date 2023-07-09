from setuptools import setup
import os.path
from configparser import ConfigParser
from multiprocessing import get_all_start_methods

# The setup file may get run once before dependencies are installed to find out what the dependencies are
# We need to encase the cython/numpy parts in a try-except to make sure that initial run succeeds.
try:
    from Cython.Build import cythonize
    import numpy
    ext_options = {"compiler_directives": {"profile": True}, "annotate": True, 'language_level':3}
    ext_modules = cythonize([
        "pynitride/core/fem.pyx",
        "pynitride/core/cython_maths.pyx",
        ], **ext_options)
    include_dirs=[numpy.get_include()]
except ModuleNotFoundError as e:
    ext_modules=None
    include_dirs=[]


#ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
setup(
    name='PyNitride',
    version='0.1.2',
    packages=['pynitride'],
    url='',
    license='',
    author='Samuel James Bader',
    author_email='samuel.james.bader@gmail.com',
    description='Python utilities for 1D band diagrams and simulation',
    ext_modules = ext_modules,
    setup_requires=['numpy','cython'],
    install_requires=['numpy','cython','matplotlib', 'scipy', 'pytest', 'pint'],
    include_dirs=include_dirs,
)

def make_default_config():
    with open("config.ini",'w') as f:
        cp=ConfigParser()
        cp.add_section("parallelism")
        cp.set("parallelism","globalthreads","cpu_count")
        cp.set("parallelism","globalprocesses","cpu_count" if 'fork' in get_all_start_methods() else '1')
        cp.set("parallelism","cextthread","1")
        cp.add_section("logging")
        cp.set("logging","level","info")
        cp.write(f)
make_default_config()
