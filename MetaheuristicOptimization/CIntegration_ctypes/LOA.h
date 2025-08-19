#ifndef LOA_H
#define LOA_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization Parameters
#define NOMAD_RATIO 0.2
#define PRIDE_SIZE 5
#define FEMALE_RATIO 0.8
#define ROAMING_RATIO 0.2
#define MATING_RATIO 0.2
#define MUTATION_PROB 0.1
#define IMMIGRATION_RATIO 0.1

// 🦁 LOA-specific Data Structure
typedef struct {
    int **prides;         // Array of prides (each pride is an array of indices)
    int *pride_sizes;     // Sizes of each pride
    int num_prides;       // Number of prides
    int *nomads;          // Array of nomad indices
    int nomad_size;       // Number of nomads
    unsigned char *genders; // Gender array (1 for female, 0 for male)
} LOAData;

// 🦁 LOA Algorithm Phases
void initialize_population_loa(Optimizer *opt, LOAData *data);
void hunting_phase(Optimizer *opt, LOAData *data, int *pride, int pride_size, ObjectiveFunction objective_function);
void move_to_safe_place_phase(Optimizer *opt, LOAData *data, int *pride, int pride_size, ObjectiveFunction objective_function);
void roaming_phase(Optimizer *opt, LOAData *data, int *pride, int pride_size, ObjectiveFunction objective_function);
void loa_mating_phase(Optimizer *opt, LOAData *data, int *pride, int pride_size, ObjectiveFunction objective_function);
void nomad_movement_phase(Optimizer *opt, LOAData *data, ObjectiveFunction objective_function);
void defense_phase(Optimizer *opt, LOAData *data, int *pride, int pride_size, ObjectiveFunction objective_function);
void immigration_phase(Optimizer *opt, LOAData *data, ObjectiveFunction objective_function);
void population_control_phase(Optimizer *opt, LOAData *data);

// 🚀 Optimization Execution
void LOA_optimize(void *optimizer, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // LOA_H
