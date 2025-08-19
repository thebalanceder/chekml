#include "JOA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Xorshift random number generator state
typedef struct {
    unsigned long long x;
} XorshiftState;

// Xorshift constants
#define JOA_INV_RAND_MAX (1.0 / (double)(1ULL << 32))

// Xorshift random number generator
static inline unsigned long long xorshift_next(XorshiftState *state) {
    unsigned long long x = state->x;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state->x = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline double xorshift_double(XorshiftState *state, double min, double max) {
    return min + (max - min) * ((double)(xorshift_next(state) >> 32) * JOA_INV_RAND_MAX);
}

// rand_double_joa implementation
double rand_double_joa(double min, double max) {
    static XorshiftState rng = { .x = 0 };
    if (rng.x == 0) {
        rng.x = (unsigned long long)time(NULL) ^ 0xDEADBEEF;
    }
    return xorshift_double(&rng, min, max);
}

// Initialize subpopulations
void initialize_subpopulations(Optimizer *opt) {
    XorshiftState rng = { .x = (unsigned long long)time(NULL) ^ 0xDEADBEEF };
    const int total_pop_size = NUM_SUBPOPULATIONS * POPULATION_SIZE_PER_SUBPOP;
    
    // Use generaloptimizer's population, update fitness only
    if (opt->population_size != total_pop_size) {
        fprintf(stderr, "Population size mismatch: expected %d, got %d\n", total_pop_size, opt->population_size);
        exit(1);
    }

    const int dim = opt->dim;
    double *bounds = opt->bounds;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total_pop_size; i++) {
        opt->population[i].fitness = INFINITY;
        // Optionally reinitialize positions if needed
        double *pos = opt->population[i].position;
        for (int k = 0; k < dim; k++) {
            pos[k] = xorshift_double(&rng, bounds[2 * k], bounds[2 * k + 1]);
        }
    }

    opt->optimize = JOA_optimize;
}

// Evaluate fitness for all subpopulations
void evaluate_subpopulations(Optimizer *opt, double (*objective_function)(double *)) {
    const int dim = opt->dim;
    Solution *pop = opt->population;
    double best_fitness = opt->best_solution.fitness;
    double *best_pos = opt->best_solution.position;

    #pragma omp parallel for schedule(static) reduction(min:best_fitness)
    for (int i = 0; i < NUM_SUBPOPULATIONS; i++) {
        int start_idx = i * POPULATION_SIZE_PER_SUBPOP;
        for (int j = 0; j < POPULATION_SIZE_PER_SUBPOP; j++) {
            int idx = start_idx + j;
            double fitness = objective_function(pop[idx].position);
            pop[idx].fitness = fitness;

            if (fitness < best_fitness) {
                best_fitness = fitness;
                #pragma omp critical
                {
                    if (fitness < opt->best_solution.fitness) {
                        opt->best_solution.fitness = fitness;
                        for (int k = 0; k < dim; k++) {
                            best_pos[k] = pop[idx].position[k];
                        }
                    }
                }
            }
        }
    }
}

// Update subpopulations
void update_subpopulations(Optimizer *opt, double *temp_direction) {
    XorshiftState rng = { .x = (unsigned long long)time(NULL) ^ 0xDEADBEEF };
    const int dim = opt->dim;
    Solution *pop = opt->population;
    double *bounds = opt->bounds;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NUM_SUBPOPULATIONS; i++) {
        int start_idx = i * POPULATION_SIZE_PER_SUBPOP;
        int other_subpop_idx = (int)(xorshift_next(&rng) % NUM_SUBPOPULATIONS);
        if (other_subpop_idx == i) {
            other_subpop_idx = (other_subpop_idx + 1) % NUM_SUBPOPULATIONS;
        }
        int other_start_idx = other_subpop_idx * POPULATION_SIZE_PER_SUBPOP;

        for (int j = 0; j < POPULATION_SIZE_PER_SUBPOP; j++) {
            int idx = start_idx + j;
            int other_ind_idx = other_start_idx + ((int)(xorshift_next(&rng) % POPULATION_SIZE_PER_SUBPOP));

            double *pos = pop[idx].position;
            double *other_pos = pop[other_ind_idx].position;
            for (int k = 0; k < dim; k++) {
                temp_direction[k] = other_pos[k] - pos[k];
                pos[k] += MUTATION_RATE * temp_direction[k];
            }
        }
    }

    enforce_bound_constraints(opt);
}

// Main optimization function
void JOA_optimize(void *opt_void, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer *)opt_void;
    const int dim = opt->dim;

    // Allocate temporary direction array
    double *temp_direction = (double *)aligned_alloc(64, dim * sizeof(double));
    if (!temp_direction) {
        fprintf(stderr, "Memory allocation failed for temp_direction\n");
        exit(1);
    }

    initialize_subpopulations(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        evaluate_subpopulations(opt, objective_function);
        update_subpopulations(opt, temp_direction);

        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Free only JOA-specific memory
    if (temp_direction) {
        free(temp_direction);
    }
}
