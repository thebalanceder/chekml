#include "IWO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed the random generator
#include <string.h>  // For memcpy
#include <math.h>    // For cos, sin in Box-Muller

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Box-Muller transform for standard normal random variable (mean=0, std=1)
double rand_normal_iwo() {
    static int has_spare = 0;
    static double spare;
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    has_spare = 1;
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 == 0.0) u1 = 1e-10; // Avoid log(0)
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    spare = r * sin(theta);
    return r * cos(theta);
}

// Initialize Population
void initialize_population_iwo(Optimizer *opt) {
    opt->population_size = INITIAL_POP_SIZE;
    opt->population = (Solution *)malloc(MAX_POP_SIZE * sizeof(Solution));
    
    for (int i = 0; i < MAX_POP_SIZE; i++) {
        opt->population[i].position = (double *)malloc(opt->dim * sizeof(double));
        opt->population[i].fitness = INFINITY;
        if (i < INITIAL_POP_SIZE) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
            }
        }
        printf("Initialized population[%d].position = %p\n", i, (void *)opt->population[i].position);
    }
    enforce_bound_constraints(opt);
}

// Evaluate Population
void evaluate_population_iwo(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: population[%d].position is NULL\n", i);
            continue;
        }
        double fitness = objective_function(opt->population[i].position);
        if (fitness != fitness) { // Check for NaN
            fprintf(stderr, "Warning: NaN fitness detected for population[%d]\n", i);
            fitness = INFINITY;
        }
        opt->population[i].fitness = fitness;
        // Update best_solution if better
        if (fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
            printf("New best fitness at population[%d]: %f\n", i, fitness);
        }
    }
}

// Update Standard Deviation
double update_standard_deviation(int iteration, int max_iter) {
    double t = (double)(max_iter - iteration) / max_iter;
    return (SIGMA_INITIAL - SIGMA_FINAL) * pow(t, EXPONENT) + SIGMA_FINAL;
}

// Reproduction Phase
void reproduction_phase(Optimizer *opt, double sigma, double (*objective_function)(double *)) {
    // Find best and worst fitness
    double best_cost = INFINITY;
    double worst_cost = -INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < best_cost) {
            best_cost = opt->population[i].fitness;
        }
        if (opt->population[i].fitness > worst_cost) {
            worst_cost = opt->population[i].fitness;
        }
    }

    // Generate seeds starting from the current population size
    int old_pop_size = opt->population_size;
    for (int i = 0; i < old_pop_size; i++) {
        // Calculate number of seeds (inverted ratio for minimization)
        int num_seeds;
        if (best_cost >= worst_cost) {
            num_seeds = MIN_SEEDS;
        } else {
            double ratio = (worst_cost - opt->population[i].fitness) / (worst_cost - best_cost);
            num_seeds = MIN_SEEDS + (int)((MAX_SEEDS - MIN_SEEDS) * ratio);
        }
        printf("Solution[%d] fitness=%f, num_seeds=%d\n", i, opt->population[i].fitness, num_seeds);

        // Generate seeds
        for (int j = 0; j < num_seeds && opt->population_size < MAX_POP_SIZE; j++) {
            int new_idx = opt->population_size;
            if (!opt->population[new_idx].position) {
                fprintf(stderr, "Error: population[%d].position is NULL\n", new_idx);
                continue;
            }
            for (int k = 0; k < opt->dim; k++) {
                opt->population[new_idx].position[k] = opt->population[i].position[k] + 
                                                     sigma * rand_normal_iwo();
                opt->population[new_idx].position[k] = fmax(opt->population[new_idx].position[k], opt->bounds[2 * k]);
                opt->population[new_idx].position[k] = fmin(opt->population[new_idx].position[k], opt->bounds[2 * k + 1]);
            }
            double fitness = objective_function(opt->population[new_idx].position);
            if (fitness != fitness) { // Check for NaN
                fprintf(stderr, "Warning: NaN fitness detected for new seed[%d]\n", new_idx);
                fitness = INFINITY;
            }
            opt->population[new_idx].fitness = fitness;
            // Update best_solution if better
            if (fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[new_idx].position[j];
                }
                printf("New best fitness at seed[%d]: %f\n", new_idx, fitness);
            }
            opt->population_size++;
        }
    }
    enforce_bound_constraints(opt);
}

// Competitive Exclusion and Sorting
void competitive_exclusion(Optimizer *opt, double (*objective_function)(double *)) {
    // Create an array of indices for sorting
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }

    // Sort indices by fitness (ascending for minimization)
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (opt->population[indices[j]].fitness > opt->population[indices[j + 1]].fitness) {
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }

    // Reorder population in-place using indices
    Solution *temp_population = (Solution *)malloc(opt->population_size * sizeof(Solution));
    for (int i = 0; i < opt->population_size; i++) {
        temp_population[i] = opt->population[indices[i]];
    }
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i] = temp_population[i];
    }
    free(temp_population);

    // Truncate to max population size
    if (opt->population_size > MAX_POP_SIZE) {
        for (int i = MAX_POP_SIZE; i < opt->population_size; i++) {
            opt->population[i].fitness = INFINITY; // Mark as unused
        }
        opt->population_size = MAX_POP_SIZE;
    }

    free(indices);

    // Verify best solution
    double best_fitness = objective_function(opt->best_solution.position);
    if (fabs(best_fitness - opt->best_solution.fitness) > 1e-10) {
        fprintf(stderr, "Warning: Best solution fitness mismatch: stored=%f, computed=%f\n", 
                opt->best_solution.fitness, best_fitness);
        opt->best_solution.fitness = best_fitness;
    }

    // Log population pointers to check for corruption
    for (int i = 0; i < opt->population_size; i++) {
        printf("After competitive_exclusion: population[%d].position = %p\n", i, (void *)opt->population[i].position);
    }
}

// Main Optimization Function
void IWO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Initialize population
    initialize_population_iwo(opt);

    // Initialize best_solution
    opt->best_solution.fitness = INFINITY;

    // Main loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate population
        evaluate_population_iwo(opt, objective_function);

        // Update standard deviation
        double sigma = update_standard_deviation(iter, opt->max_iter);

        // Reproduction phase
        reproduction_phase(opt, sigma, objective_function);

        // Competitive exclusion
        competitive_exclusion(opt, objective_function);

        // Log iteration
        printf("Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
