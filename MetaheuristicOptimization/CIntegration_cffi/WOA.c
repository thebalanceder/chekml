#include "WOA.h"
#include "generaloptimizer.h"
#include <time.h>

// Fast xorshift32_woa random number generator
unsigned int xorshift32_woa(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Fast random double in [min, max)
double fast_rand_double(double min, double max, unsigned int *rng_state) {
    unsigned int r = xorshift32_woa(rng_state);
    return min + (max - min) * ((double)r / (double)0xffffffff);
}

// Initialize the population of search agents
void initialize_positions(Optimizer *opt, unsigned int *rng_state) {
    int i, j;
    double *bounds = opt->bounds;
    Solution *pop = opt->population;
    int dim = opt->dim;

    for (i = 0; i < opt->population_size; i++) {
        double *pos = pop[i].position;
        for (j = 0; j < dim; j++) {
            double lb = bounds[2 * j];
            double ub = bounds[2 * j + 1];
            pos[j] = lb + fast_rand_double(0.0, 1.0, rng_state) * (ub - lb);
        }
        pop[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Update the leader (best solution)
void update_leader(Optimizer *opt, double (*objective_function)(double *)) {
    int i, j;
    Solution *pop = opt->population;
    Solution *best = &opt->best_solution;
    int dim = opt->dim;
    double best_fitness = best->fitness;

    for (i = 0; i < opt->population_size; i++) {
        double fitness = objective_function(pop[i].position);
        pop[i].fitness = fitness;
        if (fitness < best_fitness) {
            best_fitness = fitness;
            for (j = 0; j < dim; j++) {
                best->position[j] = pop[i].position[j];
            }
        }
    }
    best->fitness = best_fitness;
}

// Update positions of search agents
void update_positions_woa(Optimizer *opt, int t, unsigned int *rng_state) {
    double max_iter = (double)opt->max_iter;
    double a = WOA_A_INITIAL - t * (WOA_A_INITIAL / max_iter);  // Linearly decreases from 2 to 0
    double a2 = WOA_A2_INITIAL + t * ((WOA_A2_FINAL - WOA_A2_INITIAL) / max_iter);  // Linearly decreases from -1 to -2
    double b = WOA_B;
    int i, j;
    Solution *pop = opt->population;
    Solution *best = &opt->best_solution;
    int dim = opt->dim;

    for (i = 0; i < opt->population_size; i++) {
        double r1 = fast_rand_double(0.0, 1.0, rng_state);
        double r2 = fast_rand_double(0.0, 1.0, rng_state);
        double A = 2.0 * a * r1 - a;  // Eq. (2.3)
        double C = 2.0 * r2;          // Eq. (2.4)
        double l = (a2 - 1.0) * fast_rand_double(0.0, 1.0, rng_state) + 1.0;  // Parameter in Eq. (2.5)
        double p = fast_rand_double(0.0, 1.0, rng_state);  // Strategy selection
        double *pos = pop[i].position;
        double *best_pos = best->position;

        // Unroll inner loop for small dimensions (dim <= 4)
        if (dim <= 4) {
            for (j = 0; j < dim; j++) {
                if (p < 0.5) {
                    if (fabs(A) >= 1.0) {  // Search for prey (exploration)
                        int rand_idx = (int)(fast_rand_double(0.0, 1.0, rng_state) * opt->population_size);
                        double X_rand_j = pop[rand_idx].position[j];
                        double D_X_rand = fabs(C * X_rand_j - pos[j]);  // Eq. (2.7)
                        pos[j] = X_rand_j - A * D_X_rand;  // Eq. (2.8)
                    } else {  // Encircling prey (exploitation)
                        double D_Leader = fabs(C * best_pos[j] - pos[j]);  // Eq. (2.1)
                        pos[j] = best_pos[j] - A * D_Leader;  // Eq. (2.2)
                    }
                } else {  // Spiral bubble-net attack
                    double distance2Leader = fabs(best_pos[j] - pos[j]);
                    pos[j] = distance2Leader * exp(b * l) * cos(l * 2.0 * M_PI) + best_pos[j];  // Eq. (2.5)
                }
            }
        } else {
            // Standard loop for larger dimensions
            for (j = 0; j < dim; j++) {
                if (p < 0.5) {
                    if (fabs(A) >= 1.0) {  // Search for prey (exploration)
                        int rand_idx = (int)(fast_rand_double(0.0, 1.0, rng_state) * opt->population_size);
                        double X_rand_j = pop[rand_idx].position[j];
                        double D_X_rand = fabs(C * X_rand_j - pos[j]);  // Eq. (2.7)
                        pos[j] = X_rand_j - A * D_X_rand;  // Eq. (2.8)
                    } else {  // Encircling prey (exploitation)
                        double D_Leader = fabs(C * best_pos[j] - pos[j]);  // Eq. (2.1)
                        pos[j] = best_pos[j] - A * D_Leader;  // Eq. (2.2)
                    }
                } else {  // Spiral bubble-net attack
                    double distance2Leader = fabs(best_pos[j] - pos[j]);
                    pos[j] = distance2Leader * exp(b * l) * cos(l * 2.0 * M_PI) + best_pos[j];  // Eq. (2.5)
                }
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void WOA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize random number generator state
    unsigned int rng_state = (unsigned int)time(NULL) ^ 0xdeadbeef;

    // Initialize positions
    initialize_positions(opt, &rng_state);

    // Main optimization loop
    for (int t = 0; t < opt->max_iter; t++) {
        // Update leader and fitness
        update_leader(opt, objective_function);

        // Update positions
        update_positions_woa(opt, t, &rng_state);

        // Log progress (optional, can be disabled for max performance)
        printf("Iteration %d: Best Score = %f\n", t + 1, opt->best_solution.fitness);
    }
}
