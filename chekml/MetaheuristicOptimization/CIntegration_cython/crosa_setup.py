from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="crow_search_algorithm",
    ext_modules=cythonize("crosa_algorithm.pyx", annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
