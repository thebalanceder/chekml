#include "FPA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Inline random double between min and max
static inline double fpa_rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Fast normal distribution approximation (simplified for speed)
static inline double fpa_rand_normal() {
    // Use a simple sum of uniform random variables (Central Limit Theorem approximation)
    double sum = 0.0;
    for (int i = 0; i < 12; i++) {
        sum += fpa_rand_double(0.0, 1.0);
    }
    return sum - 6.0;  // Mean 0, variance 1
}

// Optimized Lévy flight step generation
static void fpa_levy_flight(double *step, int dim) {
    const double sigma = FPA_LEVY_SIGMA;
    const double inv_beta = FPA_INV_LEVY_BETA;
    for (int i = 0; i < dim; i++) {
        double u = fpa_rand_normal() * sigma;
        double v = fpa_rand_normal();
        step[i] = FPA_LEVY_STEP_SCALE * u / pow(fabs(v), inv_beta);
    }
}

// Initialize Flowers (Population)
void fpa_initialize_flowers(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        const double *bounds = opt->bounds;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = bounds[2 * j] + fpa_rand_double(0.0, 1.0) * (bounds[2 * j + 1] - bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Global Pollination Phase (uses pre-allocated step buffer)
void fpa_global_pollination_phase(Optimizer *opt, double *step_buffer) {
    for (int i = 0; i < opt->population_size; i++) {
        if (fpa_rand_double(0.0, 1.0) > FPA_SWITCH_PROB) {
            fpa_levy_flight(step_buffer, opt->dim);
            double *pos = opt->population[i].position;
            const double *best_pos = opt->best_solution.position;
            for (int j = 0; j < opt->dim; j++) {
                pos[j] += step_buffer[j] * (pos[j] - best_pos[j]);
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Local Pollination Phase
void fpa_local_pollination_phase(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        if (fpa_rand_double(0.0, 1.0) <= FPA_SWITCH_PROB) {
            int j_idx = rand() % opt->population_size;
            int k_idx = rand() % opt->population_size;
            while (j_idx == k_idx) {
                k_idx = rand() % opt->population_size;
            }
            double epsilon = fpa_rand_double(0.0, 1.0);
            double *pos = opt->population[i].position;
            const double *pos_j = opt->population[j_idx].position;
            const double *pos_k = opt->population[k_idx].position;
            for (int j = 0; j < opt->dim; j++) {
                pos[j] += epsilon * (pos_j[j] - pos_k[j]);
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void FPA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Pre-allocate step buffer for global pollination
    double *step_buffer = (double *)malloc(opt->dim * sizeof(double));
    if (!step_buffer) {
        fprintf(stderr, "Memory allocation failed for step buffer\n");
        return;
    }

    // Seed random number generator
    srand(time(NULL));

    // Initialize population
    fpa_initialize_flowers(opt);

    // Evaluate initial population
    double best_fitness = INFINITY;
    int best_idx = 0;
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness;
        if (fitness < best_fitness) {
            best_fitness = fitness;
            best_idx = i;
        }
    }
    opt->best_solution.fitness = best_fitness;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[best_idx].position[j];
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        fpa_global_pollination_phase(opt, step_buffer);
        fpa_local_pollination_phase(opt);

        // Evaluate population and update best solution
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        enforce_bound_constraints(opt);

        // Display progress every 100 iterations
        if (iter % 100 == 0) {
            printf("Iteration %d: Best Value = %f\n", iter, opt->best_solution.fitness);
        }
    }

    // Cleanup
    free(step_buffer);

    printf("Total number of evaluations: %d\n", opt->max_iter * opt->population_size);
    printf("Best solution: [");
    for (int j = 0; j < opt->dim; j++) {
        printf("%f", opt->best_solution.position[j]);
        if (j < opt->dim - 1) printf(", ");
    }
    printf("]\n");
    printf("Best value: %f\n", opt->best_solution.fitness);
}
