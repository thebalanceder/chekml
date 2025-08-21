from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="HarmonySearch",
    ext_modules=cythonize("hs_algorithm.pyx", language_level=3),
    include_dirs=[np.get_include()]
)

