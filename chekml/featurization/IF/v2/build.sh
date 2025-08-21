#!/bin/bash

cython -3 --force ineq_cython.pyx
gcc -shared -pthread -fPIC -O3 -o ineq_cython.so ineq_cython.c inequalities.c -lm -I $(python3 -c "import numpy; print(numpy.get_include())") -I $(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
