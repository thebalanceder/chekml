from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="artificial_algae_algorithm",
    ext_modules=cythonize("aaa_algorithm.pyx", compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
