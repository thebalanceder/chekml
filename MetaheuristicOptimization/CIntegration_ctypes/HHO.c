#include "HHO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Utility function to generate a random double between min and max
double rand_double(double min, double max);

// Approximate gamma function for Levy flight (simple implementation for beta=1.5)
double approx_gamma(double x) {
    // For beta=1.5, we use precomputed approximations
    if (x == 2.5) return 1.875;  // Approximation for gamma(2.5)
    if (x == 1.75) return 0.9190625;  // Approximation for gamma(1.75)
    return 1.0;  // Fallback
}

// Levy flight step generation
void levy_flight(double *step, int dim) {
    double beta = HHO_BETA;
    double sigma = pow(
        (approx_gamma(1 + beta) * sin(M_PI * beta / 2) /
         (approx_gamma((1 + beta) / 2) * beta * pow(2, (beta - 1) / 2))),
        1 / beta
    );

    for (int i = 0; i < dim; i++) {
        // Approximate normal distribution using uniform random numbers
        double u = rand_double(-sigma, sigma);
        double v = rand_double(-1.0, 1.0);
        step[i] = u / pow(fabs(v), 1 / beta);
    }
}

// Exploration Phase (Random Perching Strategies)
void exploration_phase(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        double q = rand_double(0.0, 1.0);
        int rand_hawk_index = (int)(rand_double(0.0, opt->population_size));
        
        if (q < 0.5) {
            // Perch based on other family members
            for (int j = 0; j < opt->dim; j++) {
                double r1 = rand_double(0.0, 1.0);
                double r2 = rand_double(0.0, 1.0);
                opt->population[i].position[j] = 
                    opt->population[rand_hawk_index].position[j] - 
                    r1 * fabs(opt->population[rand_hawk_index].position[j] - 2 * r2 * opt->population[i].position[j]);
            }
        } else {
            // Perch on a random tall tree
            double mean_pos[opt->dim];
            for (int j = 0; j < opt->dim; j++) {
                mean_pos[j] = 0.0;
                for (int k = 0; k < opt->population_size; k++) {
                    mean_pos[j] += opt->population[k].position[j];
                }
                mean_pos[j] /= opt->population_size;
            }
            for (int j = 0; j < opt->dim; j++) {
                double r = rand_double(0.0, 1.0);
                opt->population[i].position[j] = 
                    (opt->best_solution.position[j] - mean_pos[j]) - 
                    r * ((opt->bounds[2 * j + 1] - opt->bounds[2 * j]) * rand_double(0.0, 1.0) + opt->bounds[2 * j]);
            }
        }
        // Update fitness
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
    enforce_bound_constraints(opt);
}

// Exploitation Phase (Soft/Hard Besiege with Rapid Dives)
void exploitation_phase(Optimizer *opt, double (*objective_function)(double *)) {
    double *levy_step = (double *)malloc(opt->dim * sizeof(double));
    if (!levy_step) {
        fprintf(stderr, "Memory allocation failed for levy_step\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        double r = rand_double(0.0, 1.0);
        double jump_strength = 2.0 * (1.0 - rand_double(0.0, 1.0));
        double escaping_energy = opt->population[i].fitness;  // Use fitness as proxy for escaping energy

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
            double X1[opt->dim], X2[opt->dim];
            for (int j = 0; j < opt->dim; j++) {
                X1[j] = opt->best_solution.position[j] - 
                        escaping_energy * fabs(jump_strength * opt->best_solution.position[j] - opt->population[i].position[j]);
            }
            double X1_fitness = objective_function(X1);
            
            if (X1_fitness < opt->population[i].fitness) {
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[i].position[j] = X1[j];
                }
                opt->population[i].fitness = X1_fitness;
            } else {
                levy_flight(levy_step, opt->dim);
                for (int j = 0; j < opt->dim; j++) {
                    X2[j] = opt->best_solution.position[j] - 
                            escaping_energy * fabs(jump_strength * opt->best_solution.position[j] - opt->population[i].position[j]) + 
                            rand_double(0.0, 1.0) * levy_step[j];
                }
                double X2_fitness = objective_function(X2);
                if (X2_fitness < opt->population[i].fitness) {
                    for (int j = 0; j < opt->dim; j++) {
                        opt->population[i].position[j] = X2[j];
                    }
                    opt->population[i].fitness = X2_fitness;
                }
            }
        } else if (r < 0.5 && fabs(escaping_energy) < 0.5) {
            // Hard besiege with rapid dives
            double mean_pos[opt->dim];
            for (int j = 0; j < opt->dim; j++) {
                mean_pos[j] = 0.0;
                for (int k = 0; k < opt->population_size; k++) {
                    mean_pos[j] += opt->population[k].position[j];
                }
                mean_pos[j] /= opt->population_size;
            }
            double X1[opt->dim], X2[opt->dim];
            for (int j = 0; j < opt->dim; j++) {
                X1[j] = opt->best_solution.position[j] - 
                        escaping_energy * fabs(jump_strength * opt->best_solution.position[j] - mean_pos[j]);
            }
            double X1_fitness = objective_function(X1);
            
            if (X1_fitness < opt->population[i].fitness) {
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[i].position[j] = X1[j];
                }
                opt->population[i].fitness = X1_fitness;
            } else {
                levy_flight(levy_step, opt->dim);
                for (int j = 0; j < opt->dim; j++) {
                    X2[j] = opt->best_solution.position[j] - 
                            escaping_energy * fabs(jump_strength * opt->best_solution.position[j] - mean_pos[j]) + 
                            rand_double(0.0, 1.0) * levy_step[j];
                }
                double X2_fitness = objective_function(X2);
                if (X2_fitness < opt->population[i].fitness) {
                    for (int j = 0; j < opt->dim; j++) {
                        opt->population[i].position[j] = X2[j];
                    }
                    opt->population[i].fitness = X2_fitness;
                }
            }
        }
        // Update fitness
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
    free(levy_step);
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void HHO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    for (int t = 0; t < opt->max_iter; t++) {
        // Update fitness for all hawks and find best solution
        for (int i = 0; i < opt->population_size; i++) {
            double fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = fitness;
            if (fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        
        // Compute escaping energy
        double E1 = ENERGY_FACTOR * (1.0 - ((double)t / opt->max_iter));
        double E0 = 2.0 * rand_double(0.0, 1.0) - 1.0;
        double Escaping_Energy = E1 * E0;
        
        // Choose exploration or exploitation based on escaping energy
        if (fabs(Escaping_Energy) >= 1.0) {
            exploration_phase(opt, objective_function);
        } else {
            exploitation_phase(opt, objective_function);
        }
        
        enforce_bound_constraints(opt);
    }
}
