from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the Cython extension
extensions = [
    Extension(
        "shuffled_frog_leaping",
        ["sfl_algorithm.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    )
]

# Setup configuration
setup(
    name="shuffled_frog_leaping",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3}),
    zip_safe=False,
)
