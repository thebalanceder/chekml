from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="energy_valley_optimizer",
    ext_modules=cythonize("evo_algorithm.pyx", annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
