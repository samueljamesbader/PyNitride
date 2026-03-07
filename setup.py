from setuptools import setup
from Cython.Build import cythonize
import numpy

# All project metadata is in pyproject.toml.
# This file exists solely to define the Cython extension modules,
# which cannot be expressed in pyproject.toml declarative config.
ext_modules = cythonize(
    [
        "pynitride/core/fem.pyx",
        "pynitride/core/cython_maths.pyx",
    ],
    compiler_directives={"profile": True, "language_level": 3},
    annotate=True,
)

setup(
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
)
