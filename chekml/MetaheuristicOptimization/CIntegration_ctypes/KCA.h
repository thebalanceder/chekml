#ifndef KCA_H
#define KCA_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // Ensure malloc/free work properly
#include <stdio.h>   // For debugging/logging
#include "generaloptimizer.h"  // Include the main optimizer header

// 🔧 Optimization parameters
#define PROBABILITY_THRESHOLD 0.5
#define HALF_POPULATION_FACTOR 0.5
#define KEY_INIT_PROB 0.5
#define FITNESS_SCALE 1.0
#define BITS_PER_DIM 10  // Number of bits per dimension for binary encoding
#define SHUBERT_MIN -5.12  // Shubert function domain
#define SHUBERT_MAX 5.12

// ⚙️ KCA Constants
#define KCA_MIN_KEY_LENGTH 1
#define KCA_MAX_KEY_LENGTH 1000
#define KCA_MIN_POP_SIZE 2
#define KCA_MAX_POP_SIZE 1000
#define KCA_DEFAULT_MAX_ITER 100

// 🌊 KCA Algorithm Functions
double rand_double(double min, double max);
void kca_binary_to_continuous(int* binary_key, double* continuous_key, int key_length, int dim);
void kca_initialize_keys(Optimizer* opt, int* keys, int key_length);
void kca_evaluate_keys(Optimizer* opt, int* keys, int key_length, double (*objective_function)(double*));
void kca_calculate_probability_factor(Optimizer* opt, int* keys, int key_length, double** prob_factors);
void kca_generate_new_keys(Optimizer* opt, int* keys, int key_length, double** prob_factors, int use_kca1);
void KCA_optimize(void* optimizer, ObjectiveFunction objective_function);  // Matches Optimizer's optimize signature

#ifdef __cplusplus
}
#endif

#endif // KCA_H