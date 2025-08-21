from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="electromagnetic_field_optimization",
    ext_modules=cythonize("efo_algorithm.pyx", annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
