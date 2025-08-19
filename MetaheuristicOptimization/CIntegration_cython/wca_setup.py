from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "water_cycle_algorithm",
        ["wca_algorithm.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="water_cycle_algorithm",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3}),
)
