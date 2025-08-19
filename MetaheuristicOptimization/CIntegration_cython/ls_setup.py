from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="local_search_cython",
    ext_modules=cythonize("ls_algorithm.pyx", annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
