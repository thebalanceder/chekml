#include "LSA.h"
#include <math.h>
#include <time.h>

// Global LCG state
uint64_t lcg_state = 123456789ULL;

// Fast normal distribution approximation
inline double rand_normal(double mean, double stddev) {
    double sum = 0.0;
    sum += rand_double_lsa();
    sum += rand_double_lsa();
    sum += rand_double_lsa();
    sum += rand_double_lsa();
    double result = mean + stddev * (sum - 2.0);
    return result > RAND_NORMAL_MAX ? RAND_NORMAL_MAX : result < -RAND_NORMAL_MAX ? -RAND_NORMAL_MAX : result;
}

// Fast exponential distribution approximation
inline double rand_exponential(double lambda) {
    lambda = lambda < MIN_LAMBDA ? MIN_LAMBDA : lambda;
    double u = rand_double_lsa();
    double result = -log(u) / lambda;
    return result > RAND_EXP_MAX ? RAND_EXP_MAX : result < -RAND_EXP_MAX ? -RAND_EXP_MAX : result;
}

// Enforce bounds
inline void enforce_bounds(double* pos, const double* bounds, int dim) {
    for (int i = 0; i < dim; i++) {
        pos[i] = pos[i] < bounds[2*i] ? bounds[2*i] : pos[i] > bounds[2*i+1] ? bounds[2*i+1] : pos[i];
    }
}

// Main LSA optimization function
void LSA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    // Initialize random seed
    lcg_state = (uint64_t)time(NULL);

    // Local variables
    double directions[2];
    double temp_pos[2];
    double lb0 = opt->bounds[0], ub0 = opt->bounds[1], diff0 = ub0 - lb0;
    double lb1 = opt->bounds[2], ub1 = opt->bounds[3], diff1 = ub1 - lb1;
    int channel_time = 0;
    int best_idx = 0, worst_idx = 0;
    double prev_best_fitness = INFINITY;
    int stall_count = 0;
    int dim = 2; // Fixed for 2D problems
    double fitness[opt->population_size];

    // Initialize population
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].position[0] = lb0 + diff0 * rand_double_lsa();
        opt->population[i].position[1] = lb1 + diff1 * rand_double_lsa();
        fitness[i] = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness[i];
        if (fitness[i] < prev_best_fitness) {
            prev_best_fitness = fitness[i];
            best_idx = i;
            opt->best_solution.position[0] = opt->population[i].position[0];
            opt->best_solution.position[1] = opt->population[i].position[1];
            opt->best_solution.fitness = fitness[i];
        }
    }
    directions[0] = rand_double_lsa() * 2.0 - 1.0;
    directions[1] = rand_double_lsa() * 2.0 - 1.0;

    // Main loop
    for (int t = 0; t < opt->max_iter; t++) {
        // Evaluate fitness and find best/worst
        prev_best_fitness = fitness[best_idx];
        worst_idx = 0;
        for (int i = 0; i < opt->population_size; i++) {
            fitness[i] = objective_function(opt->population[i].position);
            opt->population[i].fitness = fitness[i];
            if (fitness[i] < fitness[best_idx]) best_idx = i;
            if (fitness[i] > fitness[worst_idx]) worst_idx = i;
        }
        if (fitness[best_idx] < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness[best_idx];
            opt->best_solution.position[0] = opt->population[best_idx].position[0];
            opt->best_solution.position[1] = opt->population[best_idx].position[1];
        }

        // Early termination
        if (fabs(fitness[best_idx] - prev_best_fitness) < 1e-8) {
            if (++stall_count >= STALL_LIMIT) break;
        } else {
            stall_count = 0;
        }

        // Channel elimination
        if (++channel_time >= MAX_CHANNEL_TIME) {
            opt->population[worst_idx].position[0] = opt->population[best_idx].position[0];
            opt->population[worst_idx].position[1] = opt->population[best_idx].position[1];
            opt->population[worst_idx].fitness = fitness[best_idx];
            channel_time = 0;
        }

        // Update directions
        temp_pos[0] = opt->population[best_idx].position[0] + directions[0] * DIRECTION_STEP * diff0;
        temp_pos[1] = opt->population[best_idx].position[1];
        double test_fitness = objective_function(temp_pos);
        if (isfinite(test_fitness) && test_fitness < fitness[best_idx]) {
            directions[0] = directions[0];
        } else {
            directions[0] = -directions[0];
        }
        temp_pos[0] = opt->population[best_idx].position[0];
        temp_pos[1] = opt->population[best_idx].position[1] + directions[1] * DIRECTION_STEP * diff1;
        test_fitness = objective_function(temp_pos);
        if (isfinite(test_fitness) && test_fitness < fitness[best_idx]) {
            directions[1] = directions[1];
        } else {
            directions[1] = -directions[1];
        }

        // Update positions
        double energy = LSA_ENERGY_FACTOR - 2.0 * exp(-5.0 * (opt->max_iter - t) / (double)opt->max_iter);
        energy = energy > 2.0 ? 2.0 : energy;
        for (int i = 0; i < opt->population_size; i++) {
            if (i == best_idx) {
                temp_pos[0] = opt->population[i].position[0] + directions[0] * fabs(rand_normal(0.0, energy));
                temp_pos[1] = opt->population[i].position[1] + directions[1] * fabs(rand_normal(0.0, energy));
            } else {
                double dist0 = opt->population[i].position[0] - opt->population[best_idx].position[0];
                double dist1 = opt->population[i].position[1] - opt->population[best_idx].position[1];
                double r0 = rand_exponential(fabs(dist0));
                double r1 = rand_exponential(fabs(dist1));
                temp_pos[0] = opt->population[i].position[0] + (dist0 < 0 ? r0 : -r0);
                temp_pos[1] = opt->population[i].position[1] + (dist1 < 0 ? r1 : -r1);
            }

            // Check validity
            if (!isfinite(temp_pos[0]) || !isfinite(temp_pos[1])) {
                temp_pos[0] = lb0 + diff0 * rand_double_lsa();
                temp_pos[1] = lb1 + diff1 * rand_double_lsa();
            }

            // Update if better
            test_fitness = objective_function(temp_pos);
            if (isfinite(test_fitness) && test_fitness < fitness[i]) {
                opt->population[i].position[0] = temp_pos[0];
                opt->population[i].position[1] = temp_pos[1];
                opt->population[i].fitness = test_fitness;
                fitness[i] = test_fitness;

                // Focking procedure
                if (rand_double_lsa() < FOCKING_PROB) {
                    double fock_pos[2];
                    fock_pos[0] = lb0 + ub0 - temp_pos[0];
                    fock_pos[1] = lb1 + ub1 - temp_pos[1];
                    double fock_fitness = objective_function(fock_pos);
                    if (isfinite(fock_fitness) && fock_fitness < fitness[i]) {
                        opt->population[i].position[0] = fock_pos[0];
                        opt->population[i].position[1] = fock_pos[1];
                        opt->population[i].fitness = fock_fitness;
                        fitness[i] = fock_fitness;
                    }
                }
            }
        }

        // Enforce bounds
        for (int i = 0; i < opt->population_size; i++) {
            enforce_bounds(opt->population[i].position, opt->bounds, dim);
        }
    }
}
