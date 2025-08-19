from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="bat_algorithm",
    ext_modules=cythonize("ba_algorithm.pyx", annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
