from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="waterwheel_plant_algorithm",
    ext_modules=cythonize("wpa_algorithm.pyx", compiler_directives={'language_level': "3"}),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
