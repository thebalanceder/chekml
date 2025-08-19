from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="teaching_learning_based_optimization",
    ext_modules=cythonize("tlbo_algorithm.pyx"),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
