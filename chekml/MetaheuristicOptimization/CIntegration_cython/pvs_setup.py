from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="population_vortex_search",
    ext_modules=cythonize(
        "pvs_algorithm.pyx",
        annotate=True,
        compiler_directives={
            'language_level': '3',
            'cdivision': True,
            'boundscheck': False,
            'wraparound': False
        }
    ),
    include_dirs=[np.get_include()],
    zip_safe=False
)
