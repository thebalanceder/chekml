from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="AFSA Optimizer",
    ext_modules=cythonize("afsa_algorithm.pyx", language_level=3),
    include_dirs=[numpy.get_include()]
)

