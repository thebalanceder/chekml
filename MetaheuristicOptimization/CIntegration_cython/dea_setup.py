from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="dolphin_echolocation",
    ext_modules=cythonize("dea_algorithm.pyx", annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
