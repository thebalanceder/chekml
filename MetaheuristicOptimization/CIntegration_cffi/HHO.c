#include "HHO.h"
#include "generaloptimizer.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Fast Xorshift RNG (period 2^128-1)
static inline unsigned int xorshift128(unsigned int *state) {
    unsigned int x = state[3];
    x ^= x << 11;
    x ^= x >> 8;
    state[3] = state[2]; state[2] = state[1]; state[1] = state[0];
    x ^= state[0] ^ (state[0] >> 19);
    state[0] = x;
    return x;
}

// Optimized random double in [min, max)
static inline double hho_rand_double(double min, double max, unsigned int *rng_state) {
    return min + (max - min) * ((double)xorshift128(rng_state) / (double)0xFFFFFFFF);
}

// Precomputed sigma for Levy flight (beta=1.5)
static const double LEVY_SIGMA = 0.696568676784; // Precomputed for beta=1.5

// Optimized Levy flight using Box-Muller transform for normal distribution
void levy_flight(double *step, int dim, unsigned int *rng_state) {
    for (int i = 0; i < dim; i += 2) {
        double r1 = hho_rand_double(0.0, 1.0, rng_state);
        double r2 = hho_rand_double(0.0, 1.0, rng_state);
        double z0 = sqrt(-2.0 * log(r1)) * cos(2.0 * M_PI * r2);
        double z1 = sqrt(-2.0 * log(r1)) * sin(2.0 * M_PI * r2);
        step[i] = LEVY_SIGMA * z0 / pow(fabs(hho_rand_double(0.0, 1.0, rng_state)), 1.0 / HHO_BETA);
        if (i + 1 < dim) {
            step[i + 1] = LEVY_SIGMA * z1 / pow(fabs(hho_rand_double(0.0, 1.0, rng_state)), 1.0 / HHO_BETA);
        }
    }
}

// Exploration Phase
void exploration_phase(Optimizer *opt, double (*objective_function)(double *)) {
    unsigned int rng_state[4] = { (unsigned int)time(NULL), 0xDEADBEEF, 0xCAFEBABE, 0x12345678 };
    double mean_pos[opt->dim];

    // Cache population mean
    memset(mean_pos, 0, opt->dim * sizeof(double));
    for (int j = 0; j < opt->dim; j++) {
        for (int k = 0; k < opt->population_size; k++) {
            mean_pos[j] += opt->population[k].position[j];
        }
        mean_pos[j] /= opt->population_size;
    }

    for (int i = 0; i < opt->population_size; i++) {
        double q = hho_rand_double(0.0, 1.0, rng_state);
        int rand_hawk_index = (int)(hho_rand_double(0.0, opt->population_size, rng_state));

        if (q < 0.5) {
            // Perch based on other family members
            for (int j = 0; j < opt->dim; j++) {
                double r1 = hho_rand_double(0.0, 1.0, rng_state);
                double r2 = hho_rand_double(0.0, 1.0, rng_state);
                opt->population[i].position[j] = 
                    opt->population[rand_hawk_index].position[j] - 
                    r1 * fabs(opt->population[rand_hawk_index].position[j] - 2 * r2 * opt->population[i].position[j]);
            }
        } else {
            // Perch on a random tall tree
            for (int j = 0; j < opt->dim; j++) {
                double r = hho_rand_double(0.0, 1.0, rng_state);
                opt->population[i].position[j] = 
                    (opt->best_solution.position[j] - mean_pos[j]) - 
                    r * ((opt->bounds[2 * j + 1] - opt->bounds[2 * j]) * hho_rand_double(0.0, 1.0, rng_state) + opt->bounds[2 * j]);
            }
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
    enforce_bound_constraints(opt);
}

// Exploitation Phase
void exploitation_phase(Optimizer *opt, double (*objective_function)(double *)) {
    unsigned int rng_state[4] = { (unsigned int)time(NULL), 0xDEADBEEF, 0xCAFEBABE, 0x12345678 };
    double levy_step[opt->dim];
    double X1[opt->dim], X2[opt->dim];
    double mean_pos[opt->dim];

    // Cache population mean
    memset(mean_pos, 0, opt->dim * sizeof(double));
    for (int j = 0; j < opt->dim; j++) {
        for (int k = 0; k < opt->population_size; k++) {
            mean_pos[j] += opt->population[k].position[j];
        }
        mean_pos[j] /= opt->population_size;
    }

    for (int i = 0; i < opt->population_size; i++) {
        double r = hho_rand_double(0.0, 1.0, rng_state);
        double jump_strength = 2.0 * (1.0 - hho_rand_double(0.0, 1.0, rng_state));
        double escaping_energy = opt->population[i].fitness;

        if (r >= 0.5 && fabs(escaping_energy) < 0.5) {
            // Hard besiege
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = 
                    opt->best_solution.position[j] - 
                    escaping_energy * fabs(opt->best_solution.position[j] - opt->population[i].position[j]);
            }
        } else if (r >= 0.5 && fabs(escaping_energy) >= 0.5) {
            // Soft besiege
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = 
                    (opt->best_solution.position[j] - opt->population[i].position[j]) - 
                    escaping_energy * fabs(jump_strength * opt->best_solution.position[j] - opt->population[i].position[j]);
            }
        } else if (r < 0.5 && fabs(escaping_energy) >= 0.5) {
            // Soft besiege with rapid dives
            for (int j = 0; j < opt->dim; j++) {
                X1[j] = opt->best_solution.position[j] - 
                        escaping_energy * fabs(jump_strength * opt->best_solution.position[j] - opt->population[i].position[j]);
            }
            double X1_fitness = objective_function(X1);
            
            if (X1_fitness < opt->population[i].fitness) {
                memcpy(opt->population[i].position, X1, opt->dim * sizeof(double));
                opt->population[i].fitness = X1_fitness;
            } else {
                levy_flight(levy_step, opt->dim, rng_state);
                for (int j = 0; j < opt->dim; j++) {
                    X2[j] = opt->best_solution.position[j] - 
                            escaping_energy * fabs(jump_strength * opt->best_solution.position[j] - opt->population[i].position[j]) + 
                            hho_rand_double(0.0, 1.0, rng_state) * levy_step[j];
                }
                double X2_fitness = objective_function(X2);
                if (X2_fitness < opt->population[i].fitness) {
                    memcpy(opt->population[i].position, X2, opt->dim * sizeof(double));
                    opt->population[i].fitness = X2_fitness;
                }
            }
        } else if (r < 0.5 && fabs(escaping_energy) < 0.5) {
            // Hard besiege with rapid dives
            for (int j = 0; j < opt->dim; j++) {
                X1[j] = opt->best_solution.position[j] - 
                        escaping_energy * fabs(jump_strength * opt->best_solution.position[j] - mean_pos[j]);
            }
            double X1_fitness = objective_function(X1);
            
            if (X1_fitness < opt->population[i].fitness) {
                memcpy(opt->population[i].position, X1, opt->dim * sizeof(double));
                opt->population[i].fitness = X1_fitness;
            } else {
                levy_flight(levy_step, opt->dim, rng_state);
                for (int j = 0; j < opt->dim; j++) {
                    X2[j] = opt->best_solution.position[j] - 
                            escaping_energy * fabs(jump_strength * opt->best_solution.position[j] - mean_pos[j]) + 
                            hho_rand_double(0.0, 1.0, rng_state) * levy_step[j];
                }
                double X2_fitness = objective_function(X2);
                if (X2_fitness < opt->population[i].fitness) {
                    memcpy(opt->population[i].position, X2, opt->dim * sizeof(double));
                    opt->population[i].fitness = X2_fitness;
                }
            }
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void HHO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    unsigned int rng_state[4] = { (unsigned int)time(NULL), 0xDEADBEEF, 0xCAFEBABE, 0x12345678 };

    for (int t = 0; t < opt->max_iter; t++) {
        // Update fitness and find best solution
        for (int i = 0; i < opt->population_size; i++) {
            double fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = fitness;
            if (fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            }
        }

        // Compute escaping energy
        double E1 = ENERGY_FACTOR * (1.0 - ((double)t / opt->max_iter));
        double E0 = 2.0 * hho_rand_double(0.0, 1.0, rng_state) - 1.0;
        double Escaping_Energy = E1 * E0;

        // Exploration or exploitation
        if (fabs(Escaping_Energy) >= 1.0) {
            exploration_phase(opt, objective_function);
        } else {
            exploitation_phase(opt, objective_function);
        }
    }
}
