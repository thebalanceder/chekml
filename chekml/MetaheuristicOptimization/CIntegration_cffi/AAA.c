#include "AAA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h> // For memcpy

// Fast random number generator (Xorshift)
static unsigned int xorshift_state = 1;
static void init_xorshift() {
    xorshift_state = (unsigned int)time(NULL);
}

static inline unsigned int xorshift32() {
    unsigned int x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

static inline double fast_rand_double(double min, double max) {
    return min + (max - min) * ((double)xorshift32() / 0xffffffffU);
}

// Initialize Population
static inline void initialize_population(Optimizer *opt) {
    double *bounds = opt->bounds;
    int dim = opt->dim;
    int pop_size = opt->population_size;
    
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            double lower = bounds[2 * j];
            double upper = bounds[2 * j + 1];
            pos[j] = lower + fast_rand_double(0.0, 1.0) * (upper - lower);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Evaluate Population
static inline void evaluate_population(Optimizer *opt, double (*objective_function)(double *)) {
    int pop_size = opt->population_size;
    for (int i = 0; i < pop_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Movement Phase with Boundary Enforcement
static inline void movement_phase(Optimizer *opt) {
    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *best_pos = opt->best_solution.position;
    double *bounds = opt->bounds;
    
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            double direction = best_pos[j] - pos[j];
            pos[j] += STEP_SIZE * direction;
            // Inline boundary enforcement
            double lower = bounds[2 * j];
            double upper = bounds[2 * j + 1];
            if (pos[j] < lower) pos[j] = lower;
            else if (pos[j] > upper) pos[j] = upper;
        }
    }
}

// Main Optimization Function
void AAA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    init_xorshift(); // Initialize fast RNG
    initialize_population(opt);
    
    double *best_pos = opt->best_solution.position;
    double *temp_pos = (double *)malloc(opt->dim * sizeof(double));
    if (!temp_pos) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness and find best
        double best_fitness = opt->best_solution.fitness;
        int pop_size = opt->population_size;
        int dim = opt->dim;
        
        for (int i = 0; i < pop_size; i++) {
            double fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = fitness;
            if (fitness < best_fitness) {
                best_fitness = fitness;
                memcpy(temp_pos, opt->population[i].position, dim * sizeof(double));
            }
        }
        
        // Update best solution
        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = best_fitness;
            memcpy(best_pos, temp_pos, dim * sizeof(double));
        }
        
        // Update population positions
        movement_phase(opt);
        
        // Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
    
    free(temp_pos);
}
