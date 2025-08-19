#include "FPA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Function to generate a random double between min and max
double fpa_rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Approximate normal distribution using Box-Muller transform
double fpa_rand_normal() {
    double u1 = fpa_rand_double(0.0, 1.0);
    double u2 = fpa_rand_double(0.0, 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Lévy flight step generation
void fpa_levy_flight(double *step, int dim) {
    double sigma = FPA_LEVY_SIGMA;  // Precomputed for beta = 1.5
    for (int i = 0; i < dim; i++) {
        double u = fpa_rand_normal() * sigma;
        double v = fpa_rand_normal();
        step[i] = FPA_LEVY_STEP_SCALE * u / pow(fabs(v), 1.0 / FPA_LEVY_BETA);
    }
}

// Initialize Flowers (Population)
void fpa_initialize_flowers(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                             fpa_rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;  // Will be updated by objective function
    }
    enforce_bound_constraints(opt);
}

// Global Pollination Phase
void fpa_global_pollination_phase(Optimizer *opt) {
    double *step = (double *)malloc(opt->dim * sizeof(double));
    if (!step) {
        fprintf(stderr, "Memory allocation failed for step\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        if (fpa_rand_double(0.0, 1.0) > FPA_SWITCH_PROB) {
            fpa_levy_flight(step, opt->dim);
            for (int j = 0; j < opt->dim; j++) {
                double delta = step[j] * (opt->population[i].position[j] - opt->best_solution.position[j]);
                opt->population[i].position[j] += delta;
            }
        }
    }
    free(step);
    enforce_bound_constraints(opt);
}

// Local Pollination Phase
void fpa_local_pollination_phase(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        if (fpa_rand_double(0.0, 1.0) <= FPA_SWITCH_PROB) {
            // Select two random flowers
            int j_idx = rand() % opt->population_size;
            int k_idx = rand() % opt->population_size;
            while (j_idx == k_idx) {
                k_idx = rand() % opt->population_size;
            }
            double epsilon = fpa_rand_double(0.0, 1.0);
            for (int j = 0; j < opt->dim; j++) {
                double diff = opt->population[j_idx].position[j] - opt->population[k_idx].position[j];
                opt->population[i].position[j] += epsilon * diff;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void FPA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    fpa_initialize_flowers(opt);

    // Evaluate initial population
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        fpa_global_pollination_phase(opt);
        fpa_local_pollination_phase(opt);

        // Evaluate population and update best solution
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            if (new_fitness < opt->population[i].fitness) {
                opt->population[i].fitness = new_fitness;
            }
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

    printf("Total number of evaluations: %d\n", opt->max_iter * opt->population_size);
    printf("Best solution: [");
    for (int j = 0; j < opt->dim; j++) {
        printf("%f", opt->best_solution.position[j]);
        if (j < opt->dim - 1) printf(", ");
    }
    printf("]\n");
    printf("Best value: %f\n", opt->best_solution.fitness);
}
