from setuptools import setup # type: ignore (not a runtime dependency)
from Cython.Build import cythonize
import numpy

ext_modules = cythonize(
    [
        "src/pynitride/core/fem.pyx",
        "src/pynitride/core/cython_maths.pyx",
    ],
    compiler_directives={
        #"profile": True, # Causes issues with debugpy
        "language_level": 3},
    annotate=True,
)

setup(
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
)
