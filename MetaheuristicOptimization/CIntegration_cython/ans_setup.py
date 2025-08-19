# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("ans_algorithm.pyx", language_level=3),
    include_dirs=[numpy.get_include()]
)

