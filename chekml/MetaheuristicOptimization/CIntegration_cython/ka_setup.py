from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='keshtel_optimizer',
    ext_modules=cythonize("ka_algorithm.pyx", language_level=3),
    include_dirs=[numpy.get_include()]
)

