#!/bin/bash

echo "Compiling GPC..."
gcc -shared -o GPC.so -fPIC GPC.c -lm

echo "Build complete!"

