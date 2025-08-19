from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="fruit_fly_optimization_algorithm",
    ext_modules=cythonize("ffo_algorithm.pyx"),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
