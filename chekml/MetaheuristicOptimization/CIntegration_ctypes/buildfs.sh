#!/bin/bash
gcc -shared -o libfeature_selection.so -fPIC feature_selection.c -lm
