#include <stdlib.h>
#include <math.h>
#include "custom_operator.h"

// Binary custom operator
void custom_operator_bits(int A, int B, int D_prev, int E_prev, int *C, int *D_next, int *E_next) {
    *C = A ^ B;
    *D_next = (A ^ B) & (A ^ D_prev);
    *E_next = (((A ^ D_prev) | E_prev) ^ ((A ^ B) & (A ^ D_prev)));
}

// Differentiable custom operator
void custom_operator_diff(float A, float B, float D_prev, float E_prev, float *C, float *D_next, float *E_next) {
    float xor(float a, float b) { return a * (1 - b) + (1 - a) * b; }
    float and(float a, float b) { return a * b; }
    float or(float a, float b) { return a + b - a * b; }

    *C = xor(A, B);
    *D_next = and(xor(A, B), xor(A, D_prev));
    *E_next = xor(or(xor(A, D_prev), E_prev), and(xor(A, B), xor(A, D_prev)));
}
