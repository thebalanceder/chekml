from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="monkey_search_optimization_cy",
    ext_modules=cythonize("mso_algorithm.pyx", compiler_directives={'language_level': "3"}),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
