from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="lightning_search_algorithm",
    ext_modules=cythonize(
        "lsa_algorithm.pyx",
        annotate=True,  # Generate HTML annotation file for inspection
        compiler_directives={'language_level': '3'}
    ),
    include_dirs=[np.get_include()],  # Include NumPy headers
    zip_safe=False,
)
