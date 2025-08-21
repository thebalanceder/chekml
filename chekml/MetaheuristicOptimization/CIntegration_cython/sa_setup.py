# setup.py

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="SimulatedAnnealing",
    ext_modules=cythonize("sa_algorithm.pyx", language_level=3),
    include_dirs=[numpy.get_include()]
)

