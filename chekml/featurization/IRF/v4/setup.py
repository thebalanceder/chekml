from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'metrics',
        ['metrics.cpp'],
        include_dirs=[
            pybind11.get_include(),
            '/usr/include',
            '/usr/include/CL',
        ],
        extra_compile_args=['-O3', '-fopenmp', '-std=c++11'],
        extra_link_args=['-fopenmp', '-lOpenCL'],
        libraries=['OpenCL'],
        library_dirs=[
            '/usr/lib',
            '/usr/lib/x86_64-linux-gnu',
        ],
    )
]

setup(
    name='metrics',
    ext_modules=ext_modules,
    zip_safe=False,
)
