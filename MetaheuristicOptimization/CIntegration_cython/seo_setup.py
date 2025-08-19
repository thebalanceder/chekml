from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="social_engineering_optimizer",
    ext_modules=cythonize("seo_algorithm.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
