from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("dra_algorithm.pyx", language_level=3),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
