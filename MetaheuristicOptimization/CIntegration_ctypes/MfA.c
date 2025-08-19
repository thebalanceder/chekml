#include "MfA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Random number generation
double rand_double(double min, double max);

// Initialize male and female populations
void mfa_initialize_populations(Optimizer *opt, double (*objective_function)(double *)) {
    // Assume opt->population is split: first half males, second half females
    int half_pop = opt->population_size / 2;
    
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
        
        // Initialize male best positions (first half)
        if (i < half_pop) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Update male population
void mfa_update_males(Optimizer *opt, double (*objective_function)(double *), double *vel_max, double *vel_min, 
                     double inertia_weight, double personal_coeff, double global_coeff1, double distance_coeff, 
                     double nuptial_dance) {
    int half_pop = opt->population_size / 2;
    
    // Temporary arrays for velocities and best positions (males only)
    double *velocities = (double *)malloc(half_pop * opt->dim * sizeof(double));
    double *best_positions = (double *)malloc(half_pop * opt->dim * sizeof(double));
    double *best_fitnesses = (double *)malloc(half_pop * sizeof(double));
    
    // Initialize velocities and copy best positions
    for (int i = 0; i < half_pop; i++) {
        for (int j = 0; j < opt->dim; j++) {
            velocities[i * opt->dim + j] = 0.0; // Initialize velocities
            best_positions[i * opt->dim + j] = opt->population[i].position[j]; // Initial best
        }
        best_fitnesses[i] = opt->population[i].fitness;
    }

    for (int i = 0; i < half_pop; i++) {
        double *pos = opt->population[i].position;
        double *vel = &velocities[i * opt->dim];
        double *best_pos = &best_positions[i * opt->dim];
        
        if (opt->population[i].fitness > opt->best_solution.fitness) {
            for (int j = 0; j < opt->dim; j++) {
                double rpbest = best_pos[j] - pos[j];
                double rgbest = opt->best_solution.position[j] - pos[j];
                vel[j] = (inertia_weight * vel[j] +
                         personal_coeff * exp(-distance_coeff * rpbest * rpbest) * rpbest +
                         global_coeff1 * exp(-distance_coeff * rgbest * rgbest) * rgbest);
            }
        } else {
            for (int j = 0; j < opt->dim; j++) {
                double e = rand_double(-1.0, 1.0);
                vel[j] = inertia_weight * vel[j] + nuptial_dance * e;
            }
        }

        // Apply velocity limits
        for (int j = 0; j < opt->dim; j++) {
            if (vel[j] < vel_min[j]) vel[j] = vel_min[j];
            if (vel[j] > vel_max[j]) vel[j] = vel_max[j];
            pos[j] += vel[j];
        }

        // Evaluate new position
        opt->population[i].fitness = objective_function(pos);
        
        // Update personal best
        if (opt->population[i].fitness < best_fitnesses[i]) {
            for (int j = 0; j < opt->dim; j++) {
                best_pos[j] = pos[j];
            }
            best_fitnesses[i] = opt->population[i].fitness;
            if (best_fitnesses[i] < opt->best_solution.fitness) {
                opt->best_solution.fitness = best_fitnesses[i];
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = pos[j];
                }
            }
        }
    }

    free(velocities);
    free(best_positions);
    free(best_fitnesses);
    enforce_bound_constraints(opt);
}

// Update female population
void mfa_update_females(Optimizer *opt, double (*objective_function)(double *), double *vel_max, double *vel_min, 
                       double inertia_weight, double global_coeff2, double distance_coeff, double random_flight) {
    int half_pop = opt->population_size / 2;
    
    // Temporary array for velocities
    double *velocities = (double *)malloc(half_pop * opt->dim * sizeof(double));
    
    // Initialize velocities
    for (int i = 0; i < half_pop; i++) {
        for (int j = 0; j < opt->dim; j++) {
            velocities[i * opt->dim + j] = 0.0;
        }
    }

    for (int i = 0; i < half_pop; i++) {
        int female_idx = i + half_pop;
        double *male_pos = opt->population[i].position;
        double *female_pos = opt->population[female_idx].position;
        double *vel = &velocities[i * opt->dim];
        
        if (opt->population[female_idx].fitness > opt->population[i].fitness) {
            for (int j = 0; j < opt->dim; j++) {
                double rmf = male_pos[j] - female_pos[j];
                vel[j] = (inertia_weight * vel[j] +
                         global_coeff2 * exp(-distance_coeff * rmf * rmf) * rmf);
            }
        } else {
            for (int j = 0; j < opt->dim; j++) {
                double e = rand_double(-1.0, 1.0);
                vel[j] = inertia_weight * vel[j] + random_flight * e;
            }
        }

        // Apply velocity limits
        for (int j = 0; j < opt->dim; j++) {
            if (vel[j] < vel_min[j]) vel[j] = vel_min[j];
            if (vel[j] > vel_max[j]) vel[j] = vel_max[j];
            female_pos[j] += vel[j];
        }

        // Evaluate new position
        opt->population[female_idx].fitness = objective_function(female_pos);
    }

    free(velocities);
    enforce_bound_constraints(opt);
}

// Crossover for mating phase
void crossover_mfa(double *male_pos, double *female_pos, double *off1, double *off2, int dim, double *bounds) {
    for (int j = 0; j < dim; j++) {
        double L = rand_double(0.0, 1.0);
        off1[j] = L * male_pos[j] + (1.0 - L) * female_pos[j];
        off2[j] = L * female_pos[j] + (1.0 - L) * male_pos[j];
        
        // Enforce bounds
        if (off1[j] < bounds[2 * j]) off1[j] = bounds[2 * j];
        if (off1[j] > bounds[2 * j + 1]) off1[j] = bounds[2 * j + 1];
        if (off2[j] < bounds[2 * j]) off2[j] = bounds[2 * j];
        if (off2[j] > bounds[2 * j + 1]) off2[j] = bounds[2 * j + 1];
    }
}

// Mating phase
void mfa_mating_phase(Optimizer *opt, double (*objective_function)(double *), double *bounds) {
    int half_pop = opt->population_size / 2;
    Solution *offspring = (Solution *)malloc(MFA_NUM_OFFSPRING * sizeof(Solution));
    
    for (int i = 0; i < MFA_NUM_OFFSPRING; i++) {
        offspring[i].position = (double *)malloc(opt->dim * sizeof(double));
    }

    for (int k = 0; k < MFA_NUM_OFFSPRING / 2; k++) {
        double *off1 = offspring[2 * k].position;
        double *off2 = offspring[2 * k + 1].position;
        crossover_mfa(opt->population[k].position, opt->population[k + half_pop].position, 
                 off1, off2, opt->dim, bounds);
        
        offspring[2 * k].fitness = objective_function(off1);
        offspring[2 * k + 1].fitness = objective_function(off2);
        
        if (offspring[2 * k].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = offspring[2 * k].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = off1[j];
            }
        }
        if (offspring[2 * k + 1].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = offspring[2 * k + 1].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = off2[j];
            }
        }
    }

    // Merge offspring into population (replace worst half)
    Solution *temp_pop = (Solution *)malloc(opt->population_size * sizeof(Solution));
    for (int i = 0; i < opt->population_size; i++) {
        temp_pop[i].position = (double *)malloc(opt->dim * sizeof(double));
        memcpy(temp_pop[i].position, opt->population[i].position, opt->dim * sizeof(double));
        temp_pop[i].fitness = opt->population[i].fitness;
    }

    // Copy offspring to population (split between males and females)
    for (int i = 0; i < MFA_NUM_OFFSPRING / 2; i++) {
        memcpy(temp_pop[half_pop - 1 - i].position, offspring[2 * i].position, opt->dim * sizeof(double));
        temp_pop[half_pop - 1 - i].fitness = offspring[2 * i].fitness;
        memcpy(temp_pop[opt->population_size - 1 - i].position, offspring[2 * i + 1].position, opt->dim * sizeof(double));
        temp_pop[opt->population_size - 1 - i].fitness = offspring[2 * i + 1].fitness;
    }

    mfa_sort_and_select(opt, temp_pop, opt->population_size);

    // Free memory
    for (int i = 0; i < opt->population_size; i++) {
        free(temp_pop[i].position);
    }
    free(temp_pop);
    for (int i = 0; i < MFA_NUM_OFFSPRING; i++) {
        free(offspring[i].position);
    }
    free(offspring);
}

// Mutation phase
void mfa_mutation_phase(Optimizer *opt, double (*objective_function)(double *), double *bounds, double mutation_rate) {
    int half_pop = opt->population_size / 2;
    Solution *mutants = (Solution *)malloc(MFA_NUM_MUTANTS * sizeof(Solution));
    
    for (int i = 0; i < MFA_NUM_MUTANTS; i++) {
        mutants[i].position = (double *)malloc(opt->dim * sizeof(double));
        int idx = rand() % opt->population_size;
        int n_var = opt->dim;
        int n_mu = (int)(mutation_rate * n_var);
        double *sigma = (double *)malloc(n_var * sizeof(double));
        
        for (int j = 0; j < n_var; j++) {
            sigma[j] = 0.1 * (bounds[2 * j + 1] - bounds[2 * j]);
            mutants[i].position[j] = opt->population[idx].position[j];
        }

        // Randomly select indices to mutate
        for (int m = 0; m < n_mu; m++) {
            int j = rand() % n_var;
            mutants[i].position[j] += sigma[j] * ((double)rand() / RAND_MAX - 0.5);
            if (mutants[i].position[j] < bounds[2 * j]) mutants[i].position[j] = bounds[2 * j];
            if (mutants[i].position[j] > bounds[2 * j + 1]) mutants[i].position[j] = bounds[2 * j + 1];
        }

        mutants[i].fitness = objective_function(mutants[i].position);
        if (mutants[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = mutants[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = mutants[i].position[j];
            }
        }
        free(sigma);
    }

    // Replace worst individuals
    Solution *temp_pop = (Solution *)malloc(opt->population_size * sizeof(Solution));
    for (int i = 0; i < opt->population_size; i++) {
        temp_pop[i].position = (double *)malloc(opt->dim * sizeof(double));
        memcpy(temp_pop[i].position, opt->population[i].position, opt->dim * sizeof(double));
        temp_pop[i].fitness = opt->population[i].fitness;
    }

    for (int i = 0; i < MFA_NUM_MUTANTS; i++) {
        memcpy(temp_pop[opt->population_size - 1 - i].position, mutants[i].position, opt->dim * sizeof(double));
        temp_pop[opt->population_size - 1 - i].fitness = mutants[i].fitness;
    }

    mfa_sort_and_select(opt, temp_pop, opt->population_size);

    // Free memory
    for (int i = 0; i < opt->population_size; i++) {
        free(temp_pop[i].position);
    }
    free(temp_pop);
    for (int i = 0; i < MFA_NUM_MUTANTS; i++) {
        free(mutants[i].position);
    }
    free(mutants);
}

// Sort population and select best individuals
void mfa_sort_and_select(Optimizer *opt, Solution *population, int pop_size) {
    // Simple bubble sort for simplicity
    for (int i = 0; i < pop_size - 1; i++) {
        for (int j = 0; j < pop_size - i - 1; j++) {
            if (population[j].fitness > population[j + 1].fitness) {
                Solution temp = population[j];
                population[j] = population[j + 1];
                population[j + 1] = temp;
            }
        }
    }

    // Copy back to opt->population (first half males, second half females)
    int half_pop = pop_size / 2;
    for (int i = 0; i < half_pop; i++) {
        memcpy(opt->population[i].position, population[i].position, opt->dim * sizeof(double));
        opt->population[i].fitness = population[i].fitness;
        memcpy(opt->population[i + half_pop].position, population[half_pop + i].position, opt->dim * sizeof(double));
        opt->population[i + half_pop].fitness = population[half_pop + i].fitness;
    }
}

// Main optimization function
void MfA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Velocity limits
    double *vel_max = (double *)malloc(opt->dim * sizeof(double));
    double *vel_min = (double *)malloc(opt->dim * sizeof(double));
    for (int j = 0; j < opt->dim; j++) {
        vel_max[j] = 0.1 * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        vel_min[j] = -vel_max[j];
    }

    // Algorithm parameters
    double inertia_weight = MFA_INERTIA_WEIGHT;
    double nuptial_dance = MFA_NUPTIAL_DANCE;
    double random_flight = MFA_RANDOM_FLIGHT;

    mfa_initialize_populations(opt, objective_function);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        mfa_update_males(opt, objective_function, vel_max, vel_min, inertia_weight, MFA_PERSONAL_COEFF, 
                        MFA_GLOBAL_COEFF1, MFA_DISTANCE_COEFF, nuptial_dance);
        mfa_update_females(opt, objective_function, vel_max, vel_min, inertia_weight, MFA_GLOBAL_COEFF2, 
                          MFA_DISTANCE_COEFF, random_flight);
        mfa_mating_phase(opt, objective_function, opt->bounds);
        mfa_mutation_phase(opt, objective_function, opt->bounds, MFA_MUTATION_RATE);

        // Update parameters
        inertia_weight *= MFA_INERTIA_DAMP;
        nuptial_dance *= MFA_DANCE_DAMP;
        random_flight *= MFA_FLIGHT_DAMP;

        printf("Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }

    free(vel_max);
    free(vel_min);
}
