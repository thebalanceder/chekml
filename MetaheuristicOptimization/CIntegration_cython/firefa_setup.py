from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="firefly_algorithm_cy",
    ext_modules=cythonize("firefa_algorithm.pyx", annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
