from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="intelligence_water_drop",
    ext_modules=cythonize(
        "iwd_algorithm.pyx",
        annotate=True,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True
        }
    ),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
