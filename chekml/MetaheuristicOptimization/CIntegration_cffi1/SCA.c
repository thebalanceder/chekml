#include "SCA.h"
#include "generaloptimizer.h"
#include <math.h>

// Fast Linear Congruential Generator (LCG) for random numbers
static unsigned long rng_state = 1;
static inline double fast_rand(void) {
    rng_state = 6364136223846793005UL * rng_state + 1442695040888963407UL;
    return (double)(rng_state >> 32) / 4294967296.0; // Normalize to [0, 1)
}

// Initialize Solutions
void initialize_solutions(Optimizer *opt) {
    // Seed RNG with a fixed value for reproducibility
    rng_state = (unsigned long)time(NULL);

    double *bounds = opt->bounds;
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double lb = bounds[2 * j];
            double ub = bounds[2 * j + 1];
            pos[j] = lb + (ub - lb) * fast_rand();
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Fast sin/cos approximations (lookup table or Taylor series could be used, but simple scaling for now)
static inline double fast_sin(double x) {
    return sin(x); // Replace with lookup table or approximation for more speed if needed
}
static inline double fast_cos(double x) {
    return cos(x); // Replace with lookup table or approximation for more speed if needed
}

// Update Position using Sine-Cosine Equations
void sca_update_position(Optimizer *opt, double r1_factor) {
    const double two_pi = 6.283185307179586;
    double *best_pos = opt->best_solution.position;

    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        // Unroll inner loop for small dimensions (e.g., dim <= 4)
        int j = 0;
        for (; j <= opt->dim - 4; j += 4) {
            double r2_0 = two_pi * fast_rand();
            double r3_0 = 2.0 * fast_rand();
            double r4_0 = fast_rand();
            double delta_0 = r3_0 * best_pos[j] - pos[j];
            pos[j] += r1_factor * (r4_0 < 0.5 ? fast_sin(r2_0) : fast_cos(r2_0)) * fabs(delta_0);

            double r2_1 = two_pi * fast_rand();
            double r3_1 = 2.0 * fast_rand();
            double r4_1 = fast_rand();
            double delta_1 = r3_1 * best_pos[j + 1] - pos[j + 1];
            pos[j + 1] += r1_factor * (r4_1 < 0.5 ? fast_sin(r2_1) : fast_cos(r2_1)) * fabs(delta_1);

            double r2_2 = two_pi * fast_rand();
            double r3_2 = 2.0 * fast_rand();
            double r4_2 = fast_rand();
            double delta_2 = r3_2 * best_pos[j + 2] - pos[j + 2];
            pos[j + 2] += r1_factor * (r4_2 < 0.5 ? fast_sin(r2_2) : fast_cos(r2_2)) * fabs(delta_2);

            double r2_3 = two_pi * fast_rand();
            double r3_3 = 2.0 * fast_rand();
            double r4_3 = fast_rand();
            double delta_3 = r3_3 * best_pos[j + 3] - pos[j + 3];
            pos[j + 3] += r1_factor * (r4_3 < 0.5 ? fast_sin(r2_3) : fast_cos(r2_3)) * fabs(delta_3);
        }
        // Handle remaining dimensions
        for (; j < opt->dim; j++) {
            double r2 = two_pi * fast_rand();
            double r3 = 2.0 * fast_rand();
            double r4 = fast_rand();
            double delta = r3 * best_pos[j] - pos[j];
            pos[j] += r1_factor * (r4 < 0.5 ? fast_sin(r2) : fast_cos(r2)) * fabs(delta);
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void SCA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize solutions
    initialize_solutions(opt);

    // Evaluate initial solutions and find best
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

    // Precompute r1 decrement
    const double r1_decrement = SCA_A / opt->max_iter;

    // Main loop
    for (int t = 0; t < opt->max_iter; t++) {
        // Compute r1
        double r1_factor = SCA_A - t * r1_decrement;

        // Update positions
        sca_update_position(opt, r1_factor);

        // Evaluate solutions and update best in one pass
        for (int i = 0; i < opt->population_size; i++) {
            double fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = fitness;
            if (fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness;
                double *pos = opt->population[i].position;
                double *best_pos = opt->best_solution.position;
                for (int j = 0; j < opt->dim; j++) {
                    best_pos[j] = pos[j];
                }
            }
        }
    }

    // Minimal output
    printf("Best solution: [");
    for (int j = 0; j < opt->dim - 1; j++) {
        printf("%f, ", opt->best_solution.position[j]);
    }
    printf("%f]\nFitness: %f\n", opt->best_solution.position[opt->dim - 1], opt->best_solution.fitness);
}
