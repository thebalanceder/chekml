from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="owl_search_algorithm",
    ext_modules=cythonize("osa_algorithm.pyx", annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
