from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="future_search",
    ext_modules=cythonize(
        "fsa_algorithm.pyx",
        annotate=True,  # Generate HTML annotation file for optimization analysis
        compiler_directives={'language_level': "3"}
    ),
    include_dirs=[np.get_include()],  # Include NumPy headers
    zip_safe=False,
)
