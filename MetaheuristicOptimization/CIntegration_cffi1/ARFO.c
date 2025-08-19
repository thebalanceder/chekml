#include "ARFO.h"
#include <string.h>
#include <stdint.h>

// Fast random number generator (Xorshift64*)
static inline unsigned long long arfo_fast_rng_next(ARFO_FastRNG *restrict rng) {
    unsigned long long x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline double arfo_fast_rand_double(ARFO_FastRNG *restrict rng, double min, double max) {
    return min + (max - min) * ((double)arfo_fast_rng_next(rng) / (double)UINT64_MAX);
}

static inline void arfo_fast_rng_init(ARFO_FastRNG *restrict rng, unsigned long long seed) {
    rng->state = seed ? seed : 88172645463325252ULL;
}

// Comparison function for qsort
static inline int compare_doubles_arfo(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static inline int compare_fitness_index(const void *a, const void *b) {
    double fa = ((FitnessIndex *)a)->fitness;
    double fb = ((FitnessIndex *)b)->fitness;
    return (fa > fb) - (fa < fb);
}

// Compute auxin concentration
static inline void calculate_auxin_concentration(Optimizer *restrict opt, double *restrict fitness, double *restrict auxin) {
    double f_min = fitness[0], f_max = fitness[0];
    for (int i = 1; i < opt->population_size; i++) {
        if (fitness[i] < f_min) f_min = fitness[i];
        if (fitness[i] > f_max) f_max = fitness[i];
    }

    double sum_normalized = 0.0;
    if (f_max == f_min) {
        double val = 1.0 / opt->population_size;
        for (int i = 0; i < opt->population_size; i++) {
            auxin[i] = val;
            sum_normalized += val;
        }
    } else {
        double inv_range = 1.0 / (f_max - f_min);
        for (int i = 0; i < opt->population_size; i++) {
            auxin[i] = (fitness[i] - f_min) * inv_range;
            sum_normalized += auxin[i];
        }
    }

    double norm_factor = AUXIN_NORMALIZATION_FACTOR / sum_normalized;
    for (int i = 0; i < opt->population_size; i++) {
        auxin[i] *= norm_factor;
    }
}

// Construct Von Neumann topology
static inline void construct_von_neumann_topology(int pop_size, int *restrict topology) {
    int rows = (int)sqrt((double)pop_size);
    int cols = (pop_size + rows - 1) / rows;
    memset(topology, -1, pop_size * 4 * sizeof(int));

    for (int i = 0; i < pop_size; i++) {
        int row = i / cols;
        int col = i % cols;
        int *t = topology + i * 4;
        if (col > 0) t[0] = i - 1; // Left
        if (col < cols - 1 && i + 1 < pop_size) t[1] = i + 1; // Right
        if (row > 0) t[2] = i - cols; // Up
        if (row < (pop_size + cols - 1) / cols - 1 && i + cols < pop_size) t[3] = i + cols; // Down
    }
}

// Regrowth Phase
void regrowth_phase(Optimizer *restrict opt, int *restrict topology, double *restrict auxin, double *restrict fitness, double *restrict auxin_sorted, ARFO_FastRNG *restrict rng) {
    calculate_auxin_concentration(opt, fitness, auxin);
    memcpy(auxin_sorted, auxin, opt->population_size * sizeof(double));
    qsort(auxin_sorted, opt->population_size, sizeof(double), compare_doubles_arfo);
    double median_auxin = auxin_sorted[opt->population_size / 2];

    for (int i = 0; i < opt->population_size; i++) {
        if (auxin[i] > median_auxin) {
            int *t = topology + i * 4;
            int valid_neighbors[4];
            int valid_count = 0;
            for (int k = 0; k < 4; k++) {
                if (t[k] >= 0) valid_neighbors[valid_count++] = t[k];
            }

            int local_best_idx = i;
            if (valid_count > 0) {
                double min_fitness = fitness[valid_neighbors[0]];
                local_best_idx = valid_neighbors[0];
                for (int k = 1; k < valid_count; k++) {
                    if (fitness[valid_neighbors[k]] < min_fitness) {
                        min_fitness = fitness[valid_neighbors[k]];
                        local_best_idx = valid_neighbors[k];
                    }
                }
            }

            double *pos_i = opt->population[i].position;
            double *pos_best = opt->population[local_best_idx].position;
            for (int j = 0; j < opt->dim; j++) {
                double rand_coeff = arfo_fast_rand_double(rng, 0.0, 1.0);
                pos_i[j] += LOCAL_INERTIA * rand_coeff * (pos_best[j] - pos_i[j]);
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Branching Phase
void branching_phase(Optimizer *restrict opt, int t, double *restrict auxin, double *restrict fitness, double *restrict new_roots, int *restrict new_root_count, FitnessIndex *restrict fitness_indices, ARFO_FastRNG *restrict rng) {
    calculate_auxin_concentration(opt, fitness, auxin);
    *new_root_count = 0;

    double std = ((opt->max_iter - t) / (double)opt->max_iter) * (INITIAL_STD - FINAL_STD) + FINAL_STD;
    for (int i = 0; i < opt->population_size; i++) {
        if (auxin[i] > BRANCHING_THRESHOLD) {
            double R1 = arfo_fast_rand_double(rng, 0.0, 1.0);
            int num_new_roots = (int)(R1 * auxin[i] * (MAX_BRANCHING - MIN_BRANCHING) + MIN_BRANCHING);
            double *base_pos = opt->population[i].position;

            for (int k = 0; k < num_new_roots && *new_root_count < opt->population_size * MAX_BRANCHING; k++) {
                double *new_root = new_roots + (*new_root_count) * opt->dim;
                for (int j = 0; j < opt->dim; j++) {
                    double val = base_pos[j] + std * arfo_fast_rand_double(rng, -1.0, 1.0);
                    new_root[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], val));
                }
                (*new_root_count)++;
            }
        }
    }

    if (*new_root_count > 0) {
        for (int i = 0; i < opt->population_size; i++) {
            fitness_indices[i].fitness = fitness[i];
            fitness_indices[i].index = i;
        }
        qsort(fitness_indices, opt->population_size, sizeof(FitnessIndex), compare_fitness_index);

        int replace_count = *new_root_count < opt->population_size ? *new_root_count : opt->population_size;
        for (int i = 0; i < replace_count; i++) {
            int worst_idx = fitness_indices[opt->population_size - 1 - i].index;
            memcpy(opt->population[worst_idx].position, new_roots + i * opt->dim, opt->dim * sizeof(double));
        }
    }
}

// Lateral Growth Phase
void lateral_growth_phase(Optimizer *restrict opt, double *restrict auxin, double *restrict fitness, double *restrict auxin_sorted, double *restrict new_roots, ARFO_FastRNG *restrict rng) {
    double median_auxin = auxin_sorted[opt->population_size / 2];

    for (int i = 0; i < opt->population_size; i++) {
        if (auxin[i] <= median_auxin) {
            double rand_length = arfo_fast_rand_double(rng, 0.0, 1.0) * MAX_ELONGATION;
            double *random_vector = new_roots + i * opt->dim;
            double norm = 0.0;
            for (int j = 0; j < opt->dim; j++) {
                random_vector[j] = arfo_fast_rand_double(rng, -1.0, 1.0);
                norm += random_vector[j] * random_vector[j];
            }
            norm = sqrt(norm);
            double inv_norm = norm > 0.0 ? 1.0 / norm : 0.0;

            double *pos = opt->population[i].position;
            for (int j = 0; j < opt->dim; j++) {
                pos[j] += rand_length * (random_vector[j] * inv_norm);
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Elimination Phase
void elimination_phase_arfo(Optimizer *restrict opt, double *restrict auxin, double *restrict fitness, double *restrict auxin_sorted) {
    calculate_auxin_concentration(opt, fitness, auxin);
    memcpy(auxin_sorted, auxin, opt->population_size * sizeof(double));
    qsort(auxin_sorted, opt->population_size, sizeof(double), compare_doubles_arfo);
    double elimination_threshold = auxin_sorted[(int)(ELIMINATION_PERCENTILE * opt->population_size / 100)];

    int new_pop_size = 0;
    for (int i = 0; i < opt->population_size; i++) {
        if (auxin[i] > elimination_threshold) {
            if (new_pop_size != i) {
                memcpy(opt->population[new_pop_size].position, opt->population[i].position, opt->dim * sizeof(double));
                opt->population[new_pop_size].fitness = fitness[i];
            }
            new_pop_size++;
        }
    }
    opt->population_size = new_pop_size;
}

// Replenish Phase
void replenish_phase(Optimizer *restrict opt, double *restrict fitness, double (*objective_function)(double *), ARFO_FastRNG *restrict rng) {
    int target_pop_size = opt->population_size; // Store original size elsewhere if needed
    while (opt->population_size < target_pop_size) {
        int i = opt->population_size;
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = opt->bounds[2 * j] + arfo_fast_rand_double(rng, 0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        fitness[i] = objective_function(pos);
        opt->population[i].fitness = fitness[i];
        opt->population_size++;
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void ARFO_optimize(Optimizer *restrict opt, double (*objective_function)(double *)) {
    ARFO_FastRNG rng;
    arfo_fast_rng_init(&rng, 88172645463325252ULL);

    double *auxin = (double *)malloc(opt->population_size * sizeof(double));
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    double *auxin_sorted = (double *)malloc(opt->population_size * sizeof(double));
    int *topology = (int *)malloc(opt->population_size * 4 * sizeof(int));
    double *new_roots = (double *)malloc(opt->population_size * MAX_BRANCHING * opt->dim * sizeof(double));
    FitnessIndex *fitness_indices = (FitnessIndex *)malloc(opt->population_size * sizeof(FitnessIndex));

    int original_pop_size = opt->population_size; // Store locally
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = opt->bounds[2 * j] + arfo_fast_rand_double(&rng, 0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        fitness[i] = objective_function(pos);
        opt->population[i].fitness = fitness[i];
        if (fitness[i] < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness[i];
            memcpy(opt->best_solution.position, pos, opt->dim * sizeof(double));
        }
        fitness_indices[i].fitness = fitness[i];
        fitness_indices[i].index = i;
    }
    qsort(fitness_indices, opt->population_size, sizeof(FitnessIndex), compare_fitness_index);

    construct_von_neumann_topology(opt->population_size, topology);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        regrowth_phase(opt, topology, auxin, fitness, auxin_sorted, &rng);
        int new_root_count;
        branching_phase(opt, iter, auxin, fitness, new_roots, &new_root_count, fitness_indices, &rng);
        lateral_growth_phase(opt, auxin, fitness, auxin_sorted, new_roots, &rng);
        elimination_phase_arfo(opt, auxin, fitness, auxin_sorted);
        replenish_phase(opt, fitness, objective_function, &rng);

        for (int i = 0; i < opt->population_size; i++) {
            double *pos = opt->population[i].position;
            fitness[i] = objective_function(pos);
            opt->population[i].fitness = fitness[i];
            if (fitness[i] < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness[i];
                memcpy(opt->best_solution.position, pos, opt->dim * sizeof(double));
            }
        }
        construct_von_neumann_topology(opt->population_size, topology);
        opt->population_size = original_pop_size;
    }

    free(auxin);
    free(fitness);
    free(auxin_sorted);
    free(topology);
    free(new_roots);
    free(fitness_indices);
}
