from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="animal_migration_optimization",
    ext_modules=cythonize("amo_algorithm.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
