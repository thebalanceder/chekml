#include "BBM.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

// 64-bit Xorshift RNG for better performance
static uint64_t xorshift_state = 1;
static void xorshift_seed(uint64_t seed) {
    xorshift_state = seed ? seed : (uint64_t)time(NULL);
}
#define XorshiftNext() ( \
    xorshift_state ^= xorshift_state >> 12, \
    xorshift_state ^= xorshift_state << 25, \
    xorshift_state ^= xorshift_state >> 27, \
    xorshift_state * 0x2545F4914F6CDD1DULL \
)

// Inline random double generation
#define RAND_DOUBLE(min, max) ((min) + ((max) - (min)) * ((double)(XorshiftNext() >> 11) / 9007199254740992.0))

// Inline gamma distribution approximation
#define RAND_GAMMA(shape, scale) (CFR_FACTOR * RAND_DOUBLE(0.0, 1.0) * 2.5)

// Initialize Bees
void initialize_bees(Optimizer *opt, double *restrict temp_buffer) {
    for (int i = 0; i < opt->population_size; i++) {
        double *restrict pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = opt->bounds[2 * j] + RAND_DOUBLE(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
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
void blend_alpha_crossover(double *restrict queen, double *restrict drone, double *restrict new_solution, int dim) {
    for (int j = 0; j < dim; j++) {
        double diff = fabs(queen[j] - drone[j]);
        double lower = fmin(queen[j], drone[j]) - BLEND_ALPHA * diff;
        double upper = fmax(queen[j], drone[j]) + BLEND_ALPHA * diff;
        new_solution[j] = lower + RAND_DOUBLE(0.0, 1.0) * (upper - lower);
    }
}

// Mating Phase
void mating_phase(Optimizer *opt, double (*objective_function)(double *), double *restrict new_solution, double *restrict temp_solution) {
    double *restrict r1_buf = (double *)malloc(opt->population_size * sizeof(double));
    double *restrict r2_buf = (double *)malloc(opt->population_size * sizeof(double));
    int *restrict crossover_buf = (int *)malloc(opt->population_size * sizeof(int));
    int *restrict drone_buf = (int *)malloc(opt->population_size * sizeof(int));
    if (!r1_buf || !r2_buf || !crossover_buf || !drone_buf) {
        fprintf(stderr, "Memory allocation failed\n");
        free(r1_buf); free(r2_buf); free(crossover_buf); free(drone_buf);
        return;
    }
    
    // Batch generate random numbers and selections
    for (int i = 0; i < opt->population_size; i++) {
        r1_buf[i] = RAND_DOUBLE(0.0, 1.0);
        r2_buf[i] = RAND_DOUBLE(0.0, 1.0);
        crossover_buf[i] = opt->dim >= 4 ? (int)(RAND_DOUBLE(0.0, 4.0)) :
                          opt->dim >= 2 ? (int)(RAND_DOUBLE(0.0, 2.0)) * 3 : 3;
        drone_buf[i] = (int)(RAND_DOUBLE(0.0, opt->population_size));
    }
    
    // Precompute velocity constant
    const double vel_factor = pow(QUEEN_FACTOR, 2.0 / 3.0) / MATING_RESISTANCE;
    
    for (int i = 0; i < opt->population_size; i++) {
        double Vi = (r1_buf[i] < DRONE_SELECTION) ? (vel_factor * r1_buf[i]) : (vel_factor * r2_buf[i]);
        int crossover_type = crossover_buf[i];
        int drone_idx = drone_buf[i];
        double *restrict drone = opt->population[drone_idx].position;
        double *restrict queen = opt->best_solution.position;
        double *restrict pos = opt->population[i].position;
        
        // Perform crossover
        if (crossover_type == 0 && opt->dim >= 2) { // One-point crossover
            int cut = 1 + (int)(RAND_DOUBLE(0.0, opt->dim - 1));
            for (int j = 0; j < opt->dim; j++) {
                temp_solution[j] = (j < cut) ? queen[j] : drone[j];
            }
        } else if (crossover_type == 1 && opt->dim >= 3) { // Two-point crossover
            int cuts[2];
            cuts[0] = (int)(RAND_DOUBLE(0.0, opt->dim - 1));
            cuts[1] = cuts[0] + 1 + (int)(RAND_DOUBLE(0.0, opt->dim - cuts[0] - 1));
            for (int j = 0; j < opt->dim; j++) {
                temp_solution[j] = (j < cuts[0] || j >= cuts[1]) ? queen[j] : drone[j];
            }
        } else if (crossover_type == 2 && opt->dim >= 4) { // Three-point crossover
            int cuts[3];
            cuts[0] = (int)(RAND_DOUBLE(0.0, opt->dim - 2));
            cuts[1] = cuts[0] + 1 + (int)(RAND_DOUBLE(0.0, opt->dim - cuts[0] - 2));
            cuts[2] = cuts[1] + 1 + (int)(RAND_DOUBLE(0.0, opt->dim - cuts[1] - 1));
            for (int j = 0; j < opt->dim; j++) {
                temp_solution[j] = (j < cuts[0] || (j >= cuts[1] && j < cuts[2])) ? queen[j] : drone[j];
            }
        } else { // Blend-alpha crossover
            blend_alpha_crossover(queen, drone, temp_solution, opt->dim);
        }
        
        // Update position and apply bounds
        for (int j = 0; j < opt->dim; j++) {
            double val = queen[j] + (temp_solution[j] - queen[j]) * Vi * RAND_DOUBLE(0.0, 1.0);
            pos[j] = val < opt->bounds[2 * j] ? opt->bounds[2 * j] :
                     val > opt->bounds[2 * j + 1] ? opt->bounds[2 * j + 1] : val;
        }
        opt->population[i].fitness = objective_function(pos);
    }
    
    free(r1_buf); free(r2_buf); free(crossover_buf); free(drone_buf);
    enforce_bound_constraints(opt);
}

// Worker Phase
void worker_phase(Optimizer *opt, double (*objective_function)(double *), double *restrict new_solution) {
    double *restrict r3_buf = (double *)malloc(opt->population_size * sizeof(double));
    double *restrict r4_buf = (double *)malloc(opt->population_size * sizeof(double));
    if (!r3_buf || !r4_buf) {
        fprintf(stderr, "Memory allocation failed\n");
        free(r3_buf); free(r4_buf);
        return;
    }
    
    // Batch generate random numbers
    for (int i = 0; i < opt->population_size; i++) {
        r3_buf[i] = RAND_DOUBLE(0.0, 1.0);
        r4_buf[i] = RAND_DOUBLE(0.0, 1.0);
    }
    
    // Precompute velocity constant
    const double vel_factor = pow(WORKER_IMPROVEMENT, 2.0 / 3.0) * 0.5;
    
    for (int i = 0; i < opt->population_size; i++) {
        double CFR = RAND_GAMMA(0.85, 2.5);
        double Vi2 = (r3_buf[i] < BROOD_DISTRIBUTION) ? (vel_factor * r3_buf[i] / CFR) :
                     (vel_factor * r4_buf[i] / CFR);
        double fitness = opt->population[i].fitness;
        double improve_factor = (opt->best_solution.fitness < fitness) ? 1.0 : -1.0;
        double *restrict pos = opt->population[i].position;
        
        // Update position and apply bounds
        for (int j = 0; j < opt->dim; j++) {
            double diff = opt->best_solution.position[j] - pos[j];
            double improve = improve_factor * diff * RAND_DOUBLE(0.0, 1.0);
            double val = opt->best_solution.position[j] + diff * Vi2 + improve;
            pos[j] = val < opt->bounds[2 * j] ? opt->bounds[2 * j] :
                     val > opt->bounds[2 * j + 1] ? opt->bounds[2 * j + 1] : val;
        }
        opt->population[i].fitness = objective_function(pos);
    }
    
    free(r3_buf); free(r4_buf);
    enforce_bound_constraints(opt);
}

// Replacement Phase (using partial selection sort)
void replacement_phase(Optimizer *opt, double (*objective_function)(double *)) {
    int worst_count = (int)(REPLACEMENT_RATIO * opt->population_size);
    int *restrict indices = (int *)malloc(opt->population_size * sizeof(int));
    if (!indices) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    
    // Initialize indices
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }
    
    // Partial selection sort for worst indices
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
        double *restrict pos = opt->population[idx].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = opt->bounds[2 * j] + RAND_DOUBLE(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[idx].fitness = objective_function(pos);
    }
    
    free(indices);
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void BBM_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize random seed
    xorshift_seed((uint64_t)time(NULL));
    
    // Allocate reusable buffers with alignment
    double *new_solution = (double *)aligned_alloc(32, opt->dim * sizeof(double));
    double *temp_solution = (double *)aligned_alloc(32, opt->dim * sizeof(double));
    if (!new_solution || !temp_solution) {
        fprintf(stderr, "Memory allocation failed\n");
        free(new_solution); free(temp_solution);
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
