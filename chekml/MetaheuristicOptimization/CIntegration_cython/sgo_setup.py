from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="squid_game_optimizer",
    ext_modules=cythonize("sgo_algorithm.pyx", compiler_directives={'language_level': "3"}),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
