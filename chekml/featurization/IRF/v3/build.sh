#!/bin/bash
gcc -shared -o metrics.so -fPIC metrics.c -lm
