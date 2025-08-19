#include "SFL.h"
#include <string.h>
#include <stdint.h>
#include <time.h>

// Fast Xorshift RNG
typedef struct {
    uint64_t state;
} XorShiftRNG;

static inline void xorshift_seed(XorShiftRNG *rng, uint64_t seed) {
    rng->state = seed ? seed : 88172645463325252ULL;
}

static inline uint64_t xorshift_next(XorShiftRNG *rng) {
    uint64_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng->state = x;
    return x;
}

static inline double xorshift_double(XorShiftRNG *rng, double min, double max) {
    return min + (max - min) * (xorshift_next(rng) / (double)UINT64_MAX);
}

// Static memory
#define MAX_DIM 100
#define MAX_POP (MEMEPLEX_SIZE * NUM_MEMEPLEXES)
static double population_positions[MAX_POP * MAX_DIM];
static double population_fitness[MAX_POP];
static double memeplex_positions[MEMEPLEX_SIZE * MAX_DIM];
static double memeplex_fitness[MEMEPLEX_SIZE];
static double subcomplex_positions[NUM_PARENTS * MAX_DIM];
static double subcomplex_fitness[NUM_PARENTS];
static int memeplex_indices[NUM_MEMEPLEXES * MEMEPLEX_SIZE];
static int parent_indices[NUM_PARENTS];
static int sub_indices[NUM_PARENTS];
static double P[MEMEPLEX_SIZE];
static double lower_bound[MAX_DIM];
static double upper_bound[MAX_DIM];
static double new_pos[MAX_DIM];

// Insertion sort
static inline void insertion_sort(double *fitness, int *indices, int n) {
    for (int i = 1; i < n; i++) {
        double key_fitness = fitness[i];
        int key_index = indices[i];
        int j = i - 1;
        while (j >= 0 && fitness[j] > key_fitness) {
            fitness[j + 1] = fitness[j];
            indices[j + 1] = indices[j];
            j--;
        }
        fitness[j + 1] = key_fitness;
        indices[j + 1] = key_index;
    }
}

// Initialize population
static inline void initialize_population(Optimizer *opt, XorShiftRNG *rng) {
    int dim = opt->dim;
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = population_positions + i * dim;
        for (int j = 0; j < dim; j++) {
            pos[j] = xorshift_double(rng, opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        population_fitness[i] = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Sort population
static inline void sort_population(Optimizer *opt, SFLContext *ctx) {
    int n = opt->population_size;
    int dim = opt->dim;
    static int indices[MAX_POP];
    static double temp_positions[MAX_POP * MAX_DIM];
    static double temp_fitness[MAX_POP];

    for (int i = 0; i < n; i++) {
        indices[i] = i;
        if (population_fitness[i] == INFINITY) {
            population_fitness[i] = ctx->objective_function(population_positions + i * dim);
        }
    }

    insertion_sort(population_fitness, indices, n);

    for (int i = 0; i < n; i++) {
        temp_fitness[i] = population_fitness[i];
        memcpy(temp_positions + i * dim, population_positions + i * dim, dim * sizeof(double));
    }
    for (int i = 0; i < n; i++) {
        population_fitness[i] = temp_fitness[indices[i]];
        memcpy(population_positions + i * dim, temp_positions + indices[i] * dim, dim * sizeof(double));
    }
}

// Check bounds
static inline int is_in_range(const double *pos, const double *bounds, int dim) {
    for (int i = 0; i < dim; i++) {
        if (pos[i] < bounds[2 * i] || pos[i] > bounds[2 * i + 1]) return 0;
    }
    return 1;
}

// Run Frog Leaping Algorithm
static inline void run_fla(Optimizer *opt, SFLContext *ctx, XorShiftRNG *rng) {
    int dim = opt->dim;

    memcpy(lower_bound, memeplex_positions, dim * sizeof(double));
    memcpy(upper_bound, memeplex_positions, dim * sizeof(double));
    for (int i = 1; i < MEMEPLEX_SIZE; i++) {
        double *pos = memeplex_positions + i * dim;
        for (int j = 0; j < dim; j++) {
            if (pos[j] < lower_bound[j]) lower_bound[j] = pos[j];
            if (pos[j] > upper_bound[j]) upper_bound[j] = pos[j];
        }
    }

    for (int iter = 0; iter < MAX_FLA_ITER; iter++) {
        for (int i = 0; i < NUM_PARENTS; i++) {
            double r = xorshift_double(rng, 0.0, 1.0);
            double cumsum = 0.0;
            for (int j = 0; j < MEMEPLEX_SIZE; j++) {
                cumsum += P[j];
                if (r <= cumsum) {
                    parent_indices[i] = j;
                    break;
                }
            }
        }

        for (int i = 0; i < NUM_PARENTS; i++) {
            int idx = parent_indices[i];
            subcomplex_fitness[i] = memeplex_fitness[idx];
            memcpy(subcomplex_positions + i * dim, memeplex_positions + idx * dim, dim * sizeof(double));
            sub_indices[i] = i;
        }

        for (int k = 0; k < NUM_OFFSPRINGS; k++) {
            insertion_sort(subcomplex_fitness, sub_indices, NUM_PARENTS);

            int worst_idx = sub_indices[NUM_PARENTS - 1];
            int best_idx = sub_indices[0];
            int improvement_step2 = 0;
            int censorship = 0;

            for (int j = 0; j < dim; j++) {
                double step = SFL_STEP_SIZE * xorshift_double(rng, 0.0, 1.0) *
                              (subcomplex_positions[best_idx * dim + j] - subcomplex_positions[worst_idx * dim + j]);
                new_pos[j] = subcomplex_positions[worst_idx * dim + j] + step;
            }

            if (is_in_range(new_pos, opt->bounds, dim)) {
                double new_fitness = ctx->objective_function(new_pos);
                if (new_fitness < subcomplex_fitness[worst_idx]) {
                    subcomplex_fitness[worst_idx] = new_fitness;
                    memcpy(subcomplex_positions + worst_idx * dim, new_pos, dim * sizeof(double));
                } else {
                    improvement_step2 = 1;
                }
            } else {
                improvement_step2 = 1;
            }

            if (improvement_step2) {
                for (int j = 0; j < dim; j++) {
                    double step = SFL_STEP_SIZE * xorshift_double(rng, 0.0, 1.0) *
                                  (opt->best_solution.position[j] - subcomplex_positions[worst_idx * dim + j]);
                    new_pos[j] = subcomplex_positions[worst_idx * dim + j] + step;
                }
                if (is_in_range(new_pos, opt->bounds, dim)) {
                    double new_fitness = ctx->objective_function(new_pos);
                    if (new_fitness < subcomplex_fitness[worst_idx]) {
                        subcomplex_fitness[worst_idx] = new_fitness;
                        memcpy(subcomplex_positions + worst_idx * dim, new_pos, dim * sizeof(double));
                    } else {
                        censorship = 1;
                    }
                } else {
                    censorship = 1;
                }
            }

            if (censorship) {
                for (int j = 0; j < dim; j++) {
                    subcomplex_positions[worst_idx * dim + j] = xorshift_double(rng, lower_bound[j], upper_bound[j]);
                }
                subcomplex_fitness[worst_idx] = ctx->objective_function(subcomplex_positions + worst_idx * dim);
            }

            for (int i = 0; i < NUM_PARENTS; i++) {
                memeplex_fitness[parent_indices[i]] = subcomplex_fitness[i];
                memcpy(memeplex_positions + parent_indices[i] * dim, subcomplex_positions + i * dim, dim * sizeof(double));
            }
        }
    }
}

// Main Optimization Function
void SFL_optimize(void *opt_ptr, double (*objective_function)(double *)) {
    Optimizer *opt = (Optimizer *)opt_ptr;
    if (!opt || !objective_function || opt->population_size != POPULATION_SIZE || opt->dim > MAX_DIM || opt->max_iter <= 0) {
        fprintf(stderr, "Invalid optimizer parameters\n");
        return;
    }

    XorShiftRNG rng;
    xorshift_seed(&rng, (uint64_t)time(NULL));

    opt->optimize = SFL_optimize;
    SFLContext ctx = { .objective_function = objective_function };

    double sum_P = 0.0;
    for (int i = 0; i < MEMEPLEX_SIZE; i++) {
        P[i] = 2.0 * (MEMEPLEX_SIZE + 1 - (i + 1)) / (MEMEPLEX_SIZE * (MEMEPLEX_SIZE + 1));
        sum_P += P[i];
    }
    for (int i = 0; i < MEMEPLEX_SIZE; i++) {
        P[i] /= sum_P;
    }

    for (int i = 0; i < NUM_MEMEPLEXES; i++) {
        for (int j = 0; j < MEMEPLEX_SIZE; j++) {
            memeplex_indices[i * MEMEPLEX_SIZE + j] = i + j * NUM_MEMEPLEXES;
        }
    }

    initialize_population(opt, &rng);
    sort_population(opt, &ctx);
    opt->best_solution.fitness = population_fitness[0];
    memcpy(opt->best_solution.position, population_positions, opt->dim * sizeof(double));

    int dim = opt->dim;
    for (int iter = 0; iter < opt->max_iter; iter++) {
        for (int m = 0; m < NUM_MEMEPLEXES; m++) {
            for (int i = 0; i < MEMEPLEX_SIZE; i++) {
                int idx = memeplex_indices[m * MEMEPLEX_SIZE + i];
                memeplex_fitness[i] = population_fitness[idx];
                memcpy(memeplex_positions + i * dim, population_positions + idx * dim, dim * sizeof(double));
            }

            run_fla(opt, &ctx, &rng);

            for (int i = 0; i < MEMEPLEX_SIZE; i++) {
                int idx = memeplex_indices[m * MEMEPLEX_SIZE + i];
                population_fitness[idx] = memeplex_fitness[i];
                memcpy(population_positions + idx * dim, memeplex_positions + i * dim, dim * sizeof(double));
            }
        }

        sort_population(opt, &ctx);
        if (population_fitness[0] < opt->best_solution.fitness) {
            opt->best_solution.fitness = population_fitness[0];
            memcpy(opt->best_solution.position, population_positions, dim * sizeof(double));
        }

        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
