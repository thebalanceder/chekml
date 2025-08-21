/* SFO.c - Ultra-optimized and Robust Implementation for SailFish Optimizer */
/* Compile with: gcc -O3 -ffast-math -march=native -funroll-loops -fopenmp SFO.c -o sfo */
#include "SFO.h"
#include "generaloptimizer.h"
#include <stdlib.h> // For malloc/free, calloc
#include <time.h>   // For time
#include <string.h> // For memcpy
#include <omp.h>

/* Thread-safe random double in [0, 1) using rand_r */
static inline double rand_double(unsigned int *seed) {
    return (double)rand_r(seed) / (double)RAND_MAX;
}

/* Initialize Sardine Population */
void initialize_sardines(Optimizer *opt, Solution *sardines, int sardine_count) {
    if (!opt || !opt->bounds || !sardines || sardine_count <= 0) return;

    double * __restrict__ bounds = opt->bounds;
    const int dim = opt->dim;

    #pragma omp parallel
    {
        unsigned int seed = (unsigned int)time(NULL) ^ omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < sardine_count; i++) {
            sardines[i].position = (double *)malloc(dim * sizeof(double));
            if (!sardines[i].position) {
                fprintf(stderr, "Memory allocation failed for sardine %d\n", i);
                exit(EXIT_FAILURE);
            }
            for (int j = 0; j < dim; j++) {
                double min = bounds[2 * j];
                double max = bounds[2 * j + 1];
                sardines[i].position[j] = min + (max - min) * rand_double(&seed);
            }
            sardines[i].fitness = INFINITY;
        }
    }
}

/* Update Sailfish Positions */
void update_sailfish(Optimizer *opt, Solution *sardines, int sardine_count, double ap) {
    if (!opt || !opt->population || !opt->best_solution.position || !sardines || sardine_count <= 0) return;

    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    double * __restrict__ bounds = opt->bounds;
    Solution * __restrict__ pop = opt->population;

    #pragma omp parallel
    {
        unsigned int seed = (unsigned int)time(NULL) ^ omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < pop_size; i++) {
            double * __restrict__ pos = pop[i].position;
            if (!pos) continue;
            double r1 = rand_double(&seed);
            int sardine_idx = (int)(rand_double(&seed) * sardine_count);
            if (sardine_idx >= sardine_count) sardine_idx = sardine_count - 1;
            double * __restrict__ sardine_pos = sardines[sardine_idx].position;
            if (!sardine_pos) continue;

            for (int j = 0; j < dim; j++) {
                double target = (r1 < ap) ? opt->best_solution.position[j] : sardine_pos[j];
                double direction = target - pos[j];
                pos[j] += SFO_ALPHA * ap * direction * rand_double(&seed);
                pos[j] = fmin(fmax(pos[j], bounds[2 * j]), bounds[2 * j + 1]);
            }
        }
    }
    enforce_bound_constraints(opt);
}

/* Update Sardine Positions */
void update_sardines(Optimizer *opt, Solution *sardines, int sardine_count) {
    if (!opt || !opt->best_solution.position || !sardines || sardine_count <= 0) return;

    const int dim = opt->dim;
    double * __restrict__ bounds = opt->bounds;

    #pragma omp parallel
    {
        unsigned int seed = (unsigned int)time(NULL) ^ omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < sardine_count; i++) {
            double * __restrict__ pos = sardines[i].position;
            if (!pos) continue;
            double r2 = rand_double(&seed);
            if (r2 < 0.5) {
                // Move toward best solution
                for (int j = 0; j < dim; j++) {
                    double direction = opt->best_solution.position[j] - pos[j];
                    pos[j] += SFO_ALPHA * direction * rand_double(&seed);
                    pos[j] = fmin(fmax(pos[j], bounds[2 * j]), bounds[2 * j + 1]);
                }
            } else {
                // Random reposition
                for (int j = 0; j < dim; j++) {
                    double min = bounds[2 * j];
                    double max = bounds[2 * j + 1];
                    pos[j] = min + (max - min) * rand_double(&seed);
                }
            }
        }
    }
    // Enforce bounds on sardines
    #pragma omp parallel for
    for (int i = 0; i < sardine_count; i++) {
        if (!sardines[i].position) continue;
        for (int j = 0; j < dim; j++) {
            sardines[i].position[j] = fmin(fmax(sardines[i].position[j], bounds[2 * j]), bounds[2 * j + 1]);
        }
    }
}

/* Update Best Solution */
void update_best_solution_sfo(Optimizer *opt, Solution *sardines, int sardine_count, double (*objective_function)(double *)) {
    if (!opt || !opt->population || !opt->best_solution.position || !sardines || !objective_function || sardine_count <= 0) return;

    double best_fitness = opt->best_solution.fitness;
    double * __restrict__ best_pos = opt->best_solution.position;
    Solution * __restrict__ pop = opt->population;
    const int pop_size = opt->population_size;
    const int dim = opt->dim;

    #pragma omp parallel
    {
        unsigned int seed = (unsigned int)time(NULL) ^ omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < pop_size; i++) {
            if (!pop[i].position) continue;
            double new_fitness = objective_function(pop[i].position);
            pop[i].fitness = new_fitness;
            #pragma omp critical
            {
                if (new_fitness < best_fitness && isfinite(new_fitness)) {
                    best_fitness = new_fitness;
                    memcpy(best_pos, pop[i].position, dim * sizeof(double));
                    opt->best_solution.fitness = best_fitness;
                }
            }
        }
        #pragma omp for
        for (int i = 0; i < sardine_count; i++) {
            if (!sardines[i].position) continue;
            double new_fitness = objective_function(sardines[i].position);
            sardines[i].fitness = new_fitness;
            #pragma omp critical
            {
                if (new_fitness < best_fitness && isfinite(new_fitness)) {
                    best_fitness = new_fitness;
                    memcpy(best_pos, sardines[i].position, dim * sizeof(double));
                    opt->best_solution.fitness = best_fitness;
                }
            }
        }
    }
}

/* Main Optimization Function */
void SFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !opt->population || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Invalid optimizer or objective function\n");
        exit(EXIT_FAILURE);
    }

    const int pop_size = opt->population_size;
    if (pop_size <= 0) {
        fprintf(stderr, "Invalid population size\n");
        exit(EXIT_FAILURE);
    }
    const int sardine_count = (int)(SFO_SARDINE_RATIO * pop_size);
    if (sardine_count <= 0) {
        fprintf(stderr, "Invalid sardine count\n");
        exit(EXIT_FAILURE);
    }
    const int max_iter = opt->max_iter;
    if (max_iter <= 0) {
        fprintf(stderr, "Invalid max iterations\n");
        exit(EXIT_FAILURE);
    }

    // Allocate sardine population
    Solution *sardines = (Solution *)calloc(sardine_count, sizeof(Solution));
    if (!sardines) {
        fprintf(stderr, "Memory allocation failed for sardines\n");
        exit(EXIT_FAILURE);
    }

    // Initialize sardines
    initialize_sardines(opt, sardines, sardine_count);

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Compute attack power (linear decay)
        double ap = SFO_AP_MAX - ((SFO_AP_MAX - SFO_AP_MIN) * iter) / max_iter;

        update_sailfish(opt, sardines, sardine_count, ap);
        update_sardines(opt, sardines, sardine_count);
        update_best_solution_sfo(opt, sardines, sardine_count, objective_function);

        // Conditional debugging output
#if ENABLE_DEBUG
        if (iter % 10 == 0) {
            printf("Iteration %d: Best Value = %.6f\n", iter + 1, opt->best_solution.fitness);
        }
#endif
    }

    // Free sardine memory
    for (int i = 0; i < sardine_count; i++) {
        if (sardines[i].position) free(sardines[i].position);
    }
    free(sardines);
}
