#!/bin/bash
nvcc -c -o DISOwithRCF.o DISOwithRCF.cu -Xcompiler -fPIC -O3 -arch=sm_75
nvcc -c -o GPC.o GPC.cu -Xcompiler -fPIC -O3 -arch=sm_75
gcc -shared -o generaloptimizer.so -fPIC -mavx2 -O3 -march=native generaloptimizer.o GPC.o DISOwithRCF.o -lm -lcudart -L/usr/local/cuda/lib64

