from setuptools import setup
from Cython.Build import cythonize
import numpy
import os.path
from configparser import ConfigParser
from multiprocessing import cpu_count


def make_default_config():
    with open("config.ini",'w') as f:
        cp=ConfigParser()
        cp.add_section("parallelism")
        cp.set("parallelism","globalthreads","cpu_count")
        cp.set("parallelism","globalprocesses","cpu_count")
        cp.set("parallelism","cextthread","1")
        cp.add_section("logging")
        cp.set("logging","level","info")
        cp.write(f)

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
        "pynitride/core/fem.pyx",
        "pynitride/core/cython_maths.pyx",
        ], **ext_options),
    requires=['numpy', 'matplotlib', 'scipy', 'pytest', 'cython', 'pint'],
    include_dirs=[numpy.get_include(),ROOT_DIR]
)
make_default_config()
