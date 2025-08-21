from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("fho_algorithm.pyx", gdb_debug=True, compiler_directives={'linetrace': True}),
    include_dirs=[np.get_include()],
    extra_compile_args=["-g"],
    extra_link_args=["-g"]
)
