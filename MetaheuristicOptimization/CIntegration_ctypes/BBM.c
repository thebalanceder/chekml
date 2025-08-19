#include "BBM.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Approximate gamma distribution for CFR (simplified uniform scaling)
double rand_gamma(double shape, double scale) {
    return CFR_FACTOR * rand_double(0.0, 1.0) * 2.5;
}

// Initialize Bees
void initialize_bees(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Queen Selection Phase
void queen_selection_phase(Optimizer *opt, double (*objective_function)(double *)) {
    double min_fitness = opt->best_solution.fitness;
    int min_idx = -1;
    
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness;
        if (fitness < min_fitness) {
            min_fitness = fitness;
            min_idx = i;
        }
    }
    
    if (min_idx >= 0) {
        opt->best_solution.fitness = min_fitness;
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = opt->population[min_idx].position[j];
        }
    }
}

// Blend-Alpha Crossover
void blend_alpha_crossover(double *queen, double *drone, double *new_solution, int dim) {
    for (int j = 0; j < dim; j++) {
        double lower = fmin(queen[j], drone[j]) - BLEND_ALPHA * fabs(queen[j] - drone[j]);
        double upper = fmax(queen[j], drone[j]) + BLEND_ALPHA * fabs(queen[j] - drone[j]);
        new_solution[j] = lower + rand_double(0.0, 1.0) * (upper - lower);
    }
}

// Mating Phase
void mating_phase(Optimizer *opt, double (*objective_function)(double *)) {
    double *new_solution = (double *)malloc(opt->dim * sizeof(double));
    double *temp_solution = (double *)malloc(opt->dim * sizeof(double));
    if (!new_solution || !temp_solution) {
        fprintf(stderr, "Memory allocation failed\n");
        free(new_solution);
        free(temp_solution);
        return;
    }
    
    for (int i = 0; i < opt->population_size; i++) {
        double r1 = rand_double(0.0, 1.0);
        double r2 = rand_double(0.0, 1.0);
        double Vi = (r1 < DRONE_SELECTION) ? 
                    (pow(QUEEN_FACTOR, 2.0 / 3.0) / MATING_RESISTANCE * r1) :
                    (pow(QUEEN_FACTOR, 2.0 / 3.0) / MATING_RESISTANCE * r2);
        
        // Select crossover type
        int crossover_type;
        if (opt->dim >= 4) {
            crossover_type = rand() % 4; // 0: one-point, 1: two-point, 2: three-point, 3: blend-alpha
        } else if (opt->dim >= 2) {
            crossover_type = (rand() % 2) * 3; // 0: one-point, 3: blend-alpha
        } else {
            crossover_type = 3; // blend-alpha only
        }
        
        // Select random drone
        int drone_idx = rand() % opt->population_size;
        double *drone = opt->population[drone_idx].position;
        double *queen = opt->best_solution.position;
        
        if (crossover_type == 0 && opt->dim >= 2) { // One-point crossover
            int cut = 1 + rand() % (opt->dim - 1);
            for (int j = 0; j < cut; j++) {
                temp_solution[j] = queen[j];
            }
            for (int j = cut; j < opt->dim; j++) {
                temp_solution[j] = drone[j];
            }
        } else if (crossover_type == 1 && opt->dim >= 3) { // Two-point crossover
            int cuts[2];
            cuts[0] = rand() % (opt->dim - 1);
            cuts[1] = cuts[0] + 1 + rand() % (opt->dim - cuts[0] - 1);
            for (int j = 0; j < opt->dim; j++) {
                if (j < cuts[0] || j >= cuts[1]) {
                    temp_solution[j] = queen[j];
                } else {
                    temp_solution[j] = drone[j];
                }
            }
        } else if (crossover_type == 2 && opt->dim >= 4) { // Three-point crossover
            int cuts[3];
            cuts[0] = rand() % (opt->dim - 2);
            cuts[1] = cuts[0] + 1 + rand() % (opt->dim - cuts[0] - 2);
            cuts[2] = cuts[1] + 1 + rand() % (opt->dim - cuts[1] - 1);
            for (int j = 0; j < opt->dim; j++) {
                if (j < cuts[0] || (j >= cuts[1] && j < cuts[2])) {
                    temp_solution[j] = queen[j];
                } else {
                    temp_solution[j] = drone[j];
                }
            }
        } else { // Blend-alpha crossover
            blend_alpha_crossover(queen, drone, temp_solution, opt->dim);
        }
        
        // Apply velocity adjustment
        for (int j = 0; j < opt->dim; j++) {
            new_solution[j] = queen[j] + (temp_solution[j] - queen[j]) * Vi * rand_double(0.0, 1.0);
            if (new_solution[j] < opt->bounds[2 * j]) new_solution[j] = opt->bounds[2 * j];
            if (new_solution[j] > opt->bounds[2 * j + 1]) new_solution[j] = opt->bounds[2 * j + 1];
        }
        
        // Update population
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = new_solution[j];
        }
        opt->population[i].fitness = objective_function(new_solution);
    }
    
    free(new_solution);
    free(temp_solution);
    enforce_bound_constraints(opt);
}

// Worker Phase
void worker_phase(Optimizer *opt, double (*objective_function)(double *)) {
    double *new_solution = (double *)malloc(opt->dim * sizeof(double));
    if (!new_solution) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    
    for (int i = 0; i < opt->population_size; i++) {
        double r3 = rand_double(0.0, 1.0);
        double r4 = rand_double(0.0, 1.0);
        double CFR = rand_gamma(0.85, 2.5);
        double Vi2 = (r3 < BROOD_DISTRIBUTION) ? 
                     (pow(WORKER_IMPROVEMENT, 2.0 / 3.0) / (2.0 * CFR) * r3) :
                     (pow(WORKER_IMPROVEMENT, 2.0 / 3.0) / (2.0 * CFR) * r4);
        
        double fitness = objective_function(opt->population[i].position);
        double improve_factor = (opt->best_solution.fitness < fitness) ? 1.0 : -1.0;
        
        for (int j = 0; j < opt->dim; j++) {
            double improve = improve_factor * (opt->best_solution.position[j] - opt->population[i].position[j]) * rand_double(0.0, 1.0);
            new_solution[j] = opt->best_solution.position[j] + 
                             (opt->best_solution.position[j] - opt->population[i].position[j]) * Vi2 + 
                             improve;
            if (new_solution[j] < opt->bounds[2 * j]) new_solution[j] = opt->bounds[2 * j];
            if (new_solution[j] > opt->bounds[2 * j + 1]) new_solution[j] = opt->bounds[2 * j + 1];
        }
        
        // Update population
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = new_solution[j];
        }
        opt->population[i].fitness = objective_function(new_solution);
    }
    
    free(new_solution);
    enforce_bound_constraints(opt);
}

// Replacement Phase
void replacement_phase(Optimizer *opt, double (*objective_function)(double *)) {
    int worst_count = (int)(REPLACEMENT_RATIO * opt->population_size);
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    
    if (!fitness || !indices) {
        fprintf(stderr, "Memory allocation failed\n");
        free(fitness);
        free(indices);
        return;
    }
    
    // Compute fitness and indices
    for (int i = 0; i < opt->population_size; i++) {
        fitness[i] = opt->population[i].fitness;
        indices[i] = i;
    }
    
    // Simple bubble sort to find worst indices
    for (int i = 0; i < worst_count; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (fitness[j] > fitness[j + 1]) {
                double temp_f = fitness[j];
                fitness[j] = fitness[j + 1];
                fitness[j + 1] = temp_f;
                int temp_idx = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp_idx;
            }
        }
    }
    
    // Replace worst solutions
    for (int i = 0; i < worst_count; i++) {
        int idx = indices[opt->population_size - 1 - i];
        for (int j = 0; j < opt->dim; j++) {
            opt->population[idx].position[j] = opt->bounds[2 * j] + 
                                              rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[idx].fitness = objective_function(opt->population[idx].position);
    }
    
    free(fitness);
    free(indices);
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void BBM_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize random seed
    srand((unsigned int)time(NULL));
    
    initialize_bees(opt);
    
    for (int iter = 0; iter < opt->max_iter; iter++) {
        queen_selection_phase(opt, objective_function);
        mating_phase(opt, objective_function);
        worker_phase(opt, objective_function);
        replacement_phase(opt, objective_function);
        
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
