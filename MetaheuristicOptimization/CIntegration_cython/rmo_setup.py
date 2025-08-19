from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="radial_movement_optimization",
    ext_modules=cythonize("rmo_algorithm.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
