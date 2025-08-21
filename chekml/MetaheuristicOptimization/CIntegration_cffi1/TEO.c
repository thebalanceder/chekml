#include "TEO.h"
#include <immintrin.h> // For SIMD intrinsics (optional)

// Main Optimization Function
void TEO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Allocate memory once, aligned to 64-byte cache line
    double *current_solution = (double *)aligned_alloc(64, opt->dim * sizeof(double));
    double *new_solution = (double *)aligned_alloc(64, opt->dim * sizeof(double));
    double current_fitness = INFINITY;
    double temperature = TEO_INITIAL_TEMPERATURE;
    double previous_best_fitness = INFINITY;
    int stagnant_iterations = 0;
    const int max_stagnant_iterations = 100;

    // Initialize single RNG instance
    TEO_XorshiftRNG rng = { (uint64_t)time(NULL) ^ 0xDEADBEEF };

    // Initialize the solution
    initialize_solution(opt, current_solution, &current_fitness, objective_function, &rng);

    // Main optimization loop (unrolled partially)
    #pragma GCC unroll 4
    for (int iter = 0; iter < opt->max_iter; iter += 4) {
        for (int k = 0; k < 4 && iter + k < opt->max_iter; k++) {
            // Generate and evaluate new solution
            perturb_solution(opt, new_solution, TEO_STEP_SIZE, &rng);
            double new_fitness = objective_function(new_solution);

            // Accept or reject
            accept_solution(opt, new_solution, new_fitness, temperature, &rng);

            // Cool temperature
            temperature *= TEO_COOLING_RATE;

            // Early termination
            if (fabs(opt->best_solution.fitness - previous_best_fitness) < 1e-6) {
                stagnant_iterations++;
                if (stagnant_iterations >= max_stagnant_iterations) {
                    goto cleanup;
                }
            } else {
                stagnant_iterations = 0;
                previous_best_fitness = opt->best_solution.fitness;
            }

            // Temperature-based convergence
            if (temperature < TEO_FINAL_TEMPERATURE) {
                goto cleanup;
            }
        }
    }

cleanup:
    free(current_solution);
    free(new_solution);
}
