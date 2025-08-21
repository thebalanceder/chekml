from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="invasive_weed_optimization",
    ext_modules=cythonize(
        "iwo_algorithm.pyx",
        annotate=True,
        language_level=3
    ),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
