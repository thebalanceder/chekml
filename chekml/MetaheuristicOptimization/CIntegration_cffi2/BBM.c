#include "BBM.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

// Fast random number generator (Xorshift for better performance than rand())
static unsigned int xorshift_state = 1;
static void xorshift_seed(unsigned int seed) {
    xorshift_state = seed ? seed : (unsigned int)time(NULL);
}
static double xorshift_double(double min, double max) {
    xorshift_state ^= xorshift_state << 13;
    xorshift_state ^= xorshift_state >> 17;
    xorshift_state ^= xorshift_state << 5;
    return min + (max - min) * ((double)xorshift_state / 4294967295.0);
}

// Function to generate a random double (fallback to rand if needed)
double rand_double_bbm(double min, double max) {
    return xorshift_double(min, max);
}

// Approximate gamma distribution for CFR (simplified uniform scaling)
double rand_gamma(double shape, double scale) {
    return CFR_FACTOR * rand_double_bbm(0.0, 1.0) * 2.5;
}

// Initialize Bees
void initialize_bees(Optimizer *opt, double *temp_buffer) {
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = opt->bounds[2 * j] + rand_double_bbm(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
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
        // Only recompute fitness if position changed significantly
        if (opt->population[i].fitness == INFINITY) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }
        if (opt->population[i].fitness < min_fitness) {
            min_fitness = opt->population[i].fitness;
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
        double diff = fabs(queen[j] - drone[j]);
        double lower = fmin(queen[j], drone[j]) - BLEND_ALPHA * diff;
        double upper = fmax(queen[j], drone[j]) + BLEND_ALPHA * diff;
        new_solution[j] = lower + rand_double_bbm(0.0, 1.0) * (upper - lower);
    }
}

// Mating Phase
void mating_phase(Optimizer *opt, double (*objective_function)(double *), double *new_solution, double *temp_solution) {
    double r1_buf[opt->population_size], r2_buf[opt->population_size];
    for (int i = 0; i < opt->population_size; i++) {
        r1_buf[i] = rand_double_bbm(0.0, 1.0);
        r2_buf[i] = rand_double_bbm(0.0, 1.0);
    }
    
    for (int i = 0; i < opt->population_size; i++) {
        double r1 = r1_buf[i], r2 = r2_buf[i];
        double Vi = (r1 < DRONE_SELECTION) ? 
                    (pow(QUEEN_FACTOR, 2.0 / 3.0) / MATING_RESISTANCE * r1) :
                    (pow(QUEEN_FACTOR, 2.0 / 3.0) / MATING_RESISTANCE * r2);
        
        // Select crossover type (simplified branching)
        int crossover_type = 3; // Default to blend-alpha
        if (opt->dim >= 4) {
            crossover_type = rand() % 4;
        } else if (opt->dim >= 2) {
            crossover_type = (rand() % 2) * 3;
        }
        
        // Select random drone
        int drone_idx = rand() % opt->population_size;
        double *drone = opt->population[drone_idx].position;
        double *queen = opt->best_solution.position;
        
        if (crossover_type == 0 && opt->dim >= 2) { // One-point crossover
            int cut = 1 + rand() % (opt->dim - 1);
            for (int j = 0; j < opt->dim; j++) {
                temp_solution[j] = (j < cut) ? queen[j] : drone[j];
            }
        } else if (crossover_type == 1 && opt->dim >= 3) { // Two-point crossover
            int cuts[2];
            cuts[0] = rand() % (opt->dim - 1);
            cuts[1] = cuts[0] + 1 + rand() % (opt->dim - cuts[0] - 1);
            for (int j = 0; j < opt->dim; j++) {
                temp_solution[j] = (j < cuts[0] || j >= cuts[1]) ? queen[j] : drone[j];
            }
        } else if (crossover_type == 2 && opt->dim >= 4) { // Three-point crossover
            int cuts[3];
            cuts[0] = rand() % (opt->dim - 2);
            cuts[1] = cuts[0] + 1 + rand() % (opt->dim - cuts[0] - 2);
            cuts[2] = cuts[1] + 1 + rand() % (opt->dim - cuts[1] - 1);
            for (int j = 0; j < opt->dim; j++) {
                temp_solution[j] = (j < cuts[0] || (j >= cuts[1] && j < cuts[2])) ? queen[j] : drone[j];
            }
        } else { // Blend-alpha crossover
            blend_alpha_crossover(queen, drone, temp_solution, opt->dim);
        }
        
        // Apply velocity adjustment and bounds in one loop
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            new_solution[j] = queen[j] + (temp_solution[j] - queen[j]) * Vi * rand_double_bbm(0.0, 1.0);
            pos[j] = new_solution[j] < opt->bounds[2 * j] ? opt->bounds[2 * j] :
                     new_solution[j] > opt->bounds[2 * j + 1] ? opt->bounds[2 * j + 1] : new_solution[j];
        }
        opt->population[i].fitness = objective_function(pos);
    }
    enforce_bound_constraints(opt);
}

// Worker Phase
void worker_phase(Optimizer *opt, double (*objective_function)(double *), double *new_solution) {
    double r3_buf[opt->population_size], r4_buf[opt->population_size];
    for (int i = 0; i < opt->population_size; i++) {
        r3_buf[i] = rand_double_bbm(0.0, 1.0);
        r4_buf[i] = rand_double_bbm(0.0, 1.0);
    }
    
    for (int i = 0; i < opt->population_size; i++) {
        double r3 = r3_buf[i], r4 = r4_buf[i];
        double CFR = rand_gamma(0.85, 2.5);
        double Vi2 = (r3 < BROOD_DISTRIBUTION) ? 
                     (pow(WORKER_IMPROVEMENT, 2.0 / 3.0) / (2.0 * CFR) * r3) :
                     (pow(WORKER_IMPROVEMENT, 2.0 / 3.0) / (2.0 * CFR) * r4);
        
        double fitness = opt->population[i].fitness;
        double improve_factor = (opt->best_solution.fitness < fitness) ? 1.0 : -1.0;
        
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double improve = improve_factor * (opt->best_solution.position[j] - pos[j]) * rand_double_bbm(0.0, 1.0);
            new_solution[j] = opt->best_solution.position[j] + 
                             (opt->best_solution.position[j] - pos[j]) * Vi2 + improve;
            pos[j] = new_solution[j] < opt->bounds[2 * j] ? opt->bounds[2 * j] :
                     new_solution[j] > opt->bounds[2 * j + 1] ? opt->bounds[2 * j + 1] : new_solution[j];
        }
        opt->population[i].fitness = objective_function(pos);
    }
    enforce_bound_constraints(opt);
}

// Replacement Phase
void replacement_phase(Optimizer *opt, double (*objective_function)(double *)) {
    int worst_count = (int)(REPLACEMENT_RATIO * opt->population_size);
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    if (!indices) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    
    // Initialize indices
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }
    
    // Selection sort for worst indices (faster than bubble for small worst_count)
    for (int i = 0; i < worst_count; i++) {
        int max_idx = i;
        for (int j = i + 1; j < opt->population_size; j++) {
            if (opt->population[indices[j]].fitness > opt->population[indices[max_idx]].fitness) {
                max_idx = j;
            }
        }
        int temp = indices[i];
        indices[i] = indices[max_idx];
        indices[max_idx] = temp;
    }
    
    // Replace worst solutions
    for (int i = 0; i < worst_count; i++) {
        int idx = indices[i];
        double *pos = opt->population[idx].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = opt->bounds[2 * j] + rand_double_bbm(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[idx].fitness = objective_function(pos);
    }
    
    free(indices);
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void BBM_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize random seed
    xorshift_seed((unsigned int)time(NULL));
    
    // Allocate reusable buffers
    double *new_solution = (double *)malloc(opt->dim * sizeof(double));
    double *temp_solution = (double *)malloc(opt->dim * sizeof(double));
    if (!new_solution || !temp_solution) {
        fprintf(stderr, "Memory allocation failed\n");
        free(new_solution);
        free(temp_solution);
        return;
    }
    
    initialize_bees(opt, new_solution);
    
    for (int iter = 0; iter < opt->max_iter; iter++) {
        queen_selection_phase(opt, objective_function);
        mating_phase(opt, objective_function, new_solution, temp_solution);
        worker_phase(opt, objective_function, new_solution);
        replacement_phase(opt, objective_function);
        
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
    
    free(new_solution);
    free(temp_solution);
}
