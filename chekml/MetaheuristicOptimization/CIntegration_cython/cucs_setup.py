from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="cuckoo_search",
    ext_modules=cythonize("cucs_algorithm.pyx", compiler_directives={'language_level': "3"}),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
