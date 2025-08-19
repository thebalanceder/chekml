from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="differential_evolution",
    ext_modules=cythonize("de_algorithm.pyx", language_level=3),
    include_dirs=[np.get_include()],
)

