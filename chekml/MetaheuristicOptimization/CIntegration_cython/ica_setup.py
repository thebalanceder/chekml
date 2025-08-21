from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="imperialist_competitive_algorithm",
    ext_modules=cythonize(
        "ica_algorithm.pyx",
        annotate=True,  # Generate HTML annotation file for optimization analysis
        compiler_directives={'language_level': "3"}
    ),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
