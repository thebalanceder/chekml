from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="particle_swarm_optimization",
    ext_modules=cythonize(
        "pso_algorithm.pyx",
        annotate=True,
        compiler_directives={'language_level': 3}
    ),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
