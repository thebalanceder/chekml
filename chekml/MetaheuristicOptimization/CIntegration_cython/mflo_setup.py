from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="moth_flame_optimization",
    ext_modules=cythonize("mflo_algorithm.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
