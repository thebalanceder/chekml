# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="guided_local_search",
    ext_modules=cythonize("gls_algorithm.pyx", compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()]
)

