#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include <stdint.h>  // ✅ For uint32_t
#include "generaloptimizer.h"  // ✅ Include the main optimizer header
#include "CulA.h"  // ✅ Include CulA.h for XorshiftState definition

// 🔧 Optimization parameters
#define RANDOMIZATION_RATE 0.3
#define POSITIVE_SELECTION_RATE 0.2
#define POSITIVE_FIELD_RATIO 0.1
#define NEGATIVE_FIELD_RATIO 0.45
#define GOLDEN_RATIO 1.618033988749895  // (1 + sqrt(5)) / 2

// ⚙️ Electromagnetic Field Constants
#define POSITIVE_FIELD_SIZE(pop_size) ((int)(POSITIVE_FIELD_RATIO * (pop_size)))
#define NEGATIVE_FIELD_START(pop_size) ((int)((1.0 - NEGATIVE_FIELD_RATIO) * (pop_size)))
#define NEUTRAL_FIELD_START(pop_size) ((int)(POSITIVE_FIELD_RATIO * (pop_size)))
#define NEUTRAL_FIELD_END(pop_size) ((int)((1.0 - NEGATIVE_FIELD_RATIO) * (pop_size)))

// 🌊 EFO Algorithm Phases
void initialize_em_population(Optimizer *opt, ObjectiveFunction obj_func);
void evaluate_and_sort_population(Optimizer *opt, ObjectiveFunction obj_func, int new_particle_index);
void generate_new_particle(Optimizer *opt, int t, double *new_particle, XorshiftState *rng, double *random_values);
void insert_particle(Optimizer *opt, double *new_particle, ObjectiveFunction obj_func, int *new_particle_index, double *temp_position);

// 🚀 Optimization Execution
void EFO_optimize(void *opt, ObjectiveFunction obj_func);

#ifdef __cplusplus
}
#endif
