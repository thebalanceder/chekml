#ifndef SEO_H
#define SEO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <math.h>
#include "generaloptimizer.h"

// Optimization parameters
#define SEO_POPULATION_SIZE 50
#define SEO_MAX_ITER 100
#define SEO_RAND_BUFFER_SIZE 4096

// Random number generator state
typedef struct {
    unsigned long long state[2];  // Xorshift128+
    double *normal_buffer;       // Precomputed normal random numbers
    int normal_index;
} SEO_RNG;

// Initialize and free RNG
void seo_rng_init(SEO_RNG *rng, unsigned long seed);
void seo_rng_free(SEO_RNG *rng);

// SEO Algorithm Phases
void social_engineering_update(Optimizer *opt, SEO_RNG *rng);

// Optimization Execution
void SEO_optimize(void *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // SEO_H
