from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy

ext_modules = [
    Extension(
        "es_algorithm",
        ["es_algorithm.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++"
    )
]

setup(
    name="es_algorithm",
    ext_modules=cythonize(ext_modules, language_level=3),
)

