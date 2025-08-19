#ifndef LOA_H
#define LOA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

#define NOMAD_RATIO 0.2f
#define PRIDE_SIZE 5
#define FEMALE_RATIO 0.8f
#define ROAMING_RATIO 0.2f
#define MATING_RATIO 0.2f
#define MUTATION_PROB 0.1f
#define IMMIGRATION_RATIO 0.1f

typedef struct {
    int *prides;
    int *pride_sizes;
    int num_prides;
    int *nomads;
    int nomad_size;
    int nomad_capacity;
    unsigned char *genders;
    double *temp_buffer;
    int *females;
    int *hunters;
    int *non_hunters;
    int *males;
    int *mating_females;
    int *nomad_females;
    int *index_buffer;
    unsigned int rng_state;
} LOAData;

static inline double loa_fast_rand(LOAData *data) {
    unsigned int x = data->rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    data->rng_state = x;
    return (double)x * 2.3283064365386963e-10;
}

static inline double loa_rand_double(LOAData *data, double min, double max) {
    return min + (max - min) * loa_fast_rand(data);
}

void loa_initialize_population(Optimizer *opt, LOAData *data, ObjectiveFunction objective_function);
void hunting_phase(Optimizer *opt, LOAData *data, int pride_idx, ObjectiveFunction objective_function);
void move_to_safe_place_phase(Optimizer *opt, LOAData *data, int pride_idx, ObjectiveFunction objective_function);
void roaming_phase(Optimizer *opt, LOAData *data, int pride_idx, ObjectiveFunction objective_function);
void loa_mating_phase(Optimizer *opt, LOAData *data, int pride_idx, ObjectiveFunction objective_function);
void nomad_movement_phase(Optimizer *opt, LOAData *data, ObjectiveFunction objective_function);
void defense_phase(Optimizer *opt, LOAData *data, int pride_idx, ObjectiveFunction objective_function);
void immigration_phase(Optimizer *opt, LOAData *data, ObjectiveFunction objective_function);
void population_control_phase(Optimizer *opt, LOAData *data);
void LOA_optimize(void *optimizer, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // LOA_H
