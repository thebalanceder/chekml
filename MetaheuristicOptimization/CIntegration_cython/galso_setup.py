from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="galactic_swarm_optimization",
    ext_modules=cythonize("galso_algorithm.pyx", annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
