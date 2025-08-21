from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="dynamic_hunting_leadership_optimization",
    ext_modules=cythonize(
        "dhlo_algorithm.pyx",
        compiler_directives={'language_level': "3"}
    ),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
