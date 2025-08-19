#include "BDFO.h"
#include "generaloptimizer.h"
#include <immintrin.h>
#include <string.h>
#include <time.h>  // ✅ For time()

// Bottom Grubbing Phase (SIMD-Optimized)
void bdfo_bottom_grubbing_phase(Optimizer *opt, double (*objective_function)(double *)) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    int half_dim = (int)(dim * BDFO_SEGMENT_FACTOR);
    if (pop_size > POP_SIZE_MAX || dim > DIM_MAX) {
        fprintf(stderr, "Population size (%d) or dimensions (%d) exceed limits\n", pop_size, dim);
        return;
    }

    // Stack-based buffers
    float perturbed1[DIM_MAX];
    float perturbed2[DIM_MAX];
    float perturbation[DIM_MAX];
    float fitness_values[POP_SIZE_MAX];
    Xorshift32 rng = { (uint32_t)time(NULL) ^ 0xDEADBEEF };

    // Cache best solution
    float best_fitness = (float)opt->best_solution.fitness;
    double *best_pos = opt->best_solution.position;

    // Evaluate all fitness values
    for (int i = 0; i < pop_size; i++) {
        fitness_values[i] = (float)objective_function(opt->population[i].position);
    }

    // Process solutions with SIMD for perturbation and adjustment
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        float current_fitness = fitness_values[i];

        // Generate perturbations (unrolled loop for small dim)
        #pragma omp simd
        for (int j = 0; j < dim; j++) {
            perturbation[j] = bdfo_rand_float(&rng, -BDFO_PERTURBATION_SCALE, BDFO_PERTURBATION_SCALE);
            perturbed1[j] = (float)pos[j];
            perturbed2[j] = (float)pos[j];
        }

        // Perturb first segment (SIMD-friendly)
        #pragma omp simd
        for (int j = 0; j < half_dim; j++) {
            perturbed1[j] += perturbation[j];
        }
        // Convert to double for objective function
        double perturbed1_d[DIM_MAX];
        for (int j = 0; j < dim; j++) perturbed1_d[j] = (double)perturbed1[j];
        float fitness1 = (float)objective_function(perturbed1_d);

        // Perturb second segment
        #pragma omp simd
        for (int j = half_dim; j < dim; j++) {
            perturbed2[j] += perturbation[j];
        }
        double perturbed2_d[DIM_MAX];
        for (int j = 0; j < dim; j++) perturbed2_d[j] = (double)perturbed2[j];
        float fitness2 = (float)objective_function(perturbed2_d);

        // Adjust solution (SIMD for vectorized updates)
        if (best_fitness != INFINITY) {
            #pragma omp simd
            for (int j = 0; j < dim; j++) {
                float adjustment = BDFO_ADJUSTMENT_RATE * ((float)best_pos[j] - (float)pos[j]);
                if (j < half_dim && fitness1 < current_fitness) {
                    pos[j] += (double)adjustment;
                } else if (j >= half_dim && fitness2 < current_fitness) {
                    pos[j] += (double)adjustment;
                }
            }
        } else {
            int random_idx = (int)bdfo_rand_float(&rng, 0, pop_size);
            double *rand_pos = opt->population[random_idx].position;
            #pragma omp simd
            for (int j = 0; j < dim; j++) {
                pos[j] += BDFO_ADJUSTMENT_RATE * ((float)rand_pos[j] - (float)pos[j]);
            }
        }
    }

    enforce_bound_constraints(opt);
}

// Exploration Phase (SIMD-Optimized)
void bdfo_exploration_phase(Optimizer *opt) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    double *bounds = opt->bounds;
    Xorshift32 rng = { (uint32_t)time(NULL) ^ 0xCAFEBABE };

    #pragma omp parallel for private(rng)
    for (int i = 0; i < pop_size; i++) {
        rng.state = (uint32_t)(time(NULL) ^ (i + 0xDEADBEEF));  // Unique seed per thread
        if (bdfo_rand_float(&rng, 0.0f, 1.0f) < BDFO_EXPLORATION_PROB) {
            double *pos = opt->population[i].position;
            #pragma omp simd
            for (int j = 0; j < dim; j++) {
                float range = (float)(bounds[2 * j + 1] - bounds[2 * j]);
                pos[j] += (double)(BDFO_EXPLORATION_FACTOR * bdfo_rand_float(&rng, -1.0f, 1.0f) * range);
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Elimination Phase (Optimized)
void bdfo_elimination_phase(Optimizer *opt) {
    int worst_count = (int)(BDFO_ELIMINATION_RATIO * opt->population_size);
    int dim = opt->dim;
    double *bounds = opt->bounds;
    Xorshift32 rng = { (uint32_t)time(NULL) ^ 0xBADF00D };

    for (int i = 0; i < worst_count; i++) {
        int idx = opt->population_size - i - 1;
        double *pos = opt->population[idx].position;
        #pragma omp simd
        for (int j = 0; j < dim; j++) {
            float lower = (float)bounds[2 * j];
            float upper = (float)bounds[2 * j + 1];
            pos[j] = (double)(lower + bdfo_rand_float(&rng, 0.0f, 1.0f) * (upper - lower));
        }
        opt->population[idx].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void BDFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (opt->population_size > POP_SIZE_MAX || opt->dim > DIM_MAX) {
        fprintf(stderr, "Population size (%d) or dimensions (%d) exceed limits\n", 
                opt->population_size, opt->dim);
        return;
    }

    int pop_size = opt->population_size;
    int max_iter = opt->max_iter;
    int dim = opt->dim;
    float fitness_values[POP_SIZE_MAX];
    Xorshift32 rng = { (uint32_t)time(NULL) };

    for (int iter = 0; iter < max_iter; iter++) {
        // Evaluate fitness and update best solution
        float best_fitness = (float)opt->best_solution.fitness;
        int best_idx = -1;

        #pragma omp parallel for
        for (int i = 0; i < pop_size; i++) {
            fitness_values[i] = (float)objective_function(opt->population[i].position);
            if (fitness_values[i] < best_fitness) {
                #pragma omp critical
                {
                    if (fitness_values[i] < best_fitness) {
                        best_fitness = fitness_values[i];
                        best_idx = i;
                    }
                }
            }
        }

        if (best_idx >= 0) {
            opt->best_solution.fitness = (double)best_fitness;
            double *best_pos = opt->best_solution.position;
            double *src_pos = opt->population[best_idx].position;
            #pragma omp simd
            for (int j = 0; j < dim; j++) {
                best_pos[j] = src_pos[j];
            }
        }

        // Execute algorithm phases
        bdfo_bottom_grubbing_phase(opt, objective_function);
        bdfo_exploration_phase(opt);
        bdfo_elimination_phase(opt);

        // Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
