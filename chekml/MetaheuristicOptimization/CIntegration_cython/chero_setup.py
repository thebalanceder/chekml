from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="chemical_reaction_optimization",
    ext_modules=cythonize("chero_algorithm.pyx", annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
