from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="cultural_algorithm",
    ext_modules=cythonize("cula_algorithm.pyx", compiler_directives={'language_level': '3'}),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
