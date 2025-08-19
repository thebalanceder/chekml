#ifndef SEO_H
#define SEO_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // Ensure malloc/free work properly
#include <stdio.h>   // For logging (optional)
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization parameters
#define SEO_POPULATION_SIZE 50
#define SEO_MAX_ITER 100
#define SEO_RAND_BUFFER_SIZE 1024  // Size of precomputed random number buffer

// Random number generator state
typedef struct {
    unsigned long state;
    double *normal_buffer;  // Precomputed normal random numbers
    int normal_index;       // Current index in normal buffer
} SEO_RNG;

// Initialize and free RNG
void seo_rng_init(SEO_RNG *rng, unsigned long seed);
void seo_rng_free(SEO_RNG *rng);

// SEO Algorithm Phases
void social_engineering_update(Optimizer *opt, SEO_RNG *rng);

// Optimization Execution
void SEO_optimize(Optimizer *opt, double (*objective_function)(double *restrict));

#ifdef __cplusplus
}
#endif

#endif // SEO_H
