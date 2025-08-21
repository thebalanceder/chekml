from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        "poa_algorithm.pyx",
        annotate=True,  # Generate HTML annotation file for optimization analysis
        compiler_directives={'language_level': "3"}  # Use Python 3 syntax
    ),
    include_dirs=[np.get_include()],  # Include NumPy headers
)
