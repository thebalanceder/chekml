from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="seeker_optimization_algorithm",
    ext_modules=cythonize("soa_algorithm.pyx", annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
