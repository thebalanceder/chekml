from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='artificial_bee_colony',
    ext_modules=cythonize('abc_algorithm.pyx', annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
