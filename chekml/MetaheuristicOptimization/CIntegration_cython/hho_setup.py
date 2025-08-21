from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="harris_hawks_optimization",
    ext_modules=cythonize("hho_algorithm.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
