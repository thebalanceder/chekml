from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="bbo_algorithm",
    ext_modules=cythonize(
        "bbo_algorithm.pyx",
        compiler_directives={'language_level': "3"},
        annotate=True
    ),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
