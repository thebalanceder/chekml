from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="thermal_exchange_optimizer",
    ext_modules=cythonize(
        "teo_algorithm.pyx",
        compiler_directives={'language_level': 3},
        annotate=True  # Generate HTML annotation file for optimization analysis
    ),
    include_dirs=[np.get_include()],  # Include NumPy headers
    zip_safe=False,
)
