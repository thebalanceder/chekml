#include "MfA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Fast random number generator (xorshift for better performance)
static unsigned int xorshift_state = 1;
void xorshift_seed_mfa(unsigned int seed) {
    xorshift_state = seed ? seed : (unsigned int)time(NULL);
}
unsigned int xorshift32() {
    unsigned int x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}
double rand_double_mfa(double min, double max) {
    return min + (max - min) * ((double)xorshift32() / 0xffffffffu);
}

// Initialize male and female populations
void mfa_initialize_populations(Optimizer *opt, double (*objective_function)(double *)) {
    int half_pop = opt->population_size / 2;
    
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = rand_double_mfa(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = objective_function(pos);
        
        if (i < half_pop && opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, pos, opt->dim * sizeof(double));
        }
    }
    enforce_bound_constraints(opt);
}

// Update male population
void mfa_update_males(Optimizer *opt, double (*objective_function)(double *), double *vel_max, double *vel_min, 
                     double inertia_weight, double personal_coeff, double global_coeff1, double distance_coeff, 
                     double nuptial_dance) {
    int half_pop = opt->population_size / 2;
    
    // Persistent storage for velocities and best positions
    static double *velocities = NULL;
    static double *best_positions = NULL;
    static double *best_fitnesses = NULL;
    static int allocated_size = 0;
    
    if (allocated_size < half_pop * opt->dim) {
        free(velocities);
        free(best_positions);
        free(best_fitnesses);
        velocities = (double *)calloc(half_pop * opt->dim, sizeof(double));
        best_positions = (double *)malloc(half_pop * opt->dim * sizeof(double));
        best_fitnesses = (double *)malloc(half_pop * sizeof(double));
        allocated_size = half_pop * opt->dim;
    }
    
    // Initialize best positions and fitnesses
    for (int i = 0; i < half_pop; i++) {
        memcpy(&best_positions[i * opt->dim], opt->population[i].position, opt->dim * sizeof(double));
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
                vel[j] = fmax(vel_min[j], fmin(vel_max[j], vel[j]));
                pos[j] += vel[j];
            }
        } else {
            for (int j = 0; j < opt->dim; j++) {
                vel[j] = inertia_weight * vel[j] + nuptial_dance * rand_double_mfa(-1.0, 1.0);
                vel[j] = fmax(vel_min[j], fmin(vel_max[j], vel[j]));
                pos[j] += vel[j];
            }
        }

        opt->population[i].fitness = objective_function(pos);
        
        if (opt->population[i].fitness < best_fitnesses[i]) {
            memcpy(best_pos, pos, opt->dim * sizeof(double));
            best_fitnesses[i] = opt->population[i].fitness;
            if (best_fitnesses[i] < opt->best_solution.fitness) {
                opt->best_solution.fitness = best_fitnesses[i];
                memcpy(opt->best_solution.position, pos, opt->dim * sizeof(double));
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Update female population
void mfa_update_females(Optimizer *opt, double (*objective_function)(double *), double *vel_max, double *vel_min, 
                       double inertia_weight, double global_coeff2, double distance_coeff, double random_flight) {
    int half_pop = opt->population_size / 2;
    
    static double *velocities = NULL;
    static int allocated_size = 0;
    
    if (allocated_size < half_pop * opt->dim) {
        free(velocities);
        velocities = (double *)calloc(half_pop * opt->dim, sizeof(double));
        allocated_size = half_pop * opt->dim;
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
                vel[j] = fmax(vel_min[j], fmin(vel_max[j], vel[j]));
                female_pos[j] += vel[j];
            }
        } else {
            for (int j = 0; j < opt->dim; j++) {
                vel[j] = inertia_weight * vel[j] + random_flight * rand_double_mfa(-1.0, 1.0);
                vel[j] = fmax(vel_min[j], fmin(vel_max[j], vel[j]));
                female_pos[j] += vel[j];
            }
        }

        opt->population[female_idx].fitness = objective_function(female_pos);
    }
    enforce_bound_constraints(opt);
}

// Inline crossover for mating phase
static inline void crossover(double *male_pos, double *female_pos, double *off1, double *off2, int dim, double *bounds) {
    for (int j = 0; j < dim; j++) {
        double L = rand_double_mfa(0.0, 1.0);
        off1[j] = L * male_pos[j] + (1.0 - L) * female_pos[j];
        off2[j] = L * female_pos[j] + (1.0 - L) * male_pos[j];
        off1[j] = fmax(bounds[2 * j], fmin(bounds[2 * j + 1], off1[j]));
        off2[j] = fmax(bounds[2 * j], fmin(bounds[2 * j + 1], off2[j]));
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
        crossover(opt->population[k].position, opt->population[k + half_pop].position, 
                  off1, off2, opt->dim, bounds);
        
        offspring[2 * k].fitness = objective_function(off1);
        offspring[2 * k + 1].fitness = objective_function(off2);
        
        if (offspring[2 * k].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = offspring[2 * k].fitness;
            memcpy(opt->best_solution.position, off1, opt->dim * sizeof(double));
        }
        if (offspring[2 * k + 1].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = offspring[2 * k + 1].fitness;
            memcpy(opt->best_solution.position, off2, opt->dim * sizeof(double));
        }
    }

    Solution *temp_pop = (Solution *)malloc(opt->population_size * sizeof(Solution));
    for (int i = 0; i < opt->population_size; i++) {
        temp_pop[i].position = (double *)malloc(opt->dim * sizeof(double));
        memcpy(temp_pop[i].position, opt->population[i].position, opt->dim * sizeof(double));
        temp_pop[i].fitness = opt->population[i].fitness;
    }

    for (int i = 0; i < MFA_NUM_OFFSPRING / 2; i++) {
        memcpy(temp_pop[half_pop - 1 - i].position, offspring[2 * i].position, opt->dim * sizeof(double));
        temp_pop[half_pop - 1 - i].fitness = offspring[2 * i].fitness;
        memcpy(temp_pop[opt->population_size - 1 - i].position, offspring[2 * i + 1].position, opt->dim * sizeof(double));
        temp_pop[opt->population_size - 1 - i].fitness = offspring[2 * i + 1].fitness;
    }

    mfa_sort_and_select(opt, temp_pop, opt->population_size);

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
    int n_var = opt->dim;
    int n_mu = (int)(mutation_rate * n_var);
    double *sigma = (double *)malloc(n_var * sizeof(double));
    for (int j = 0; j < n_var; j++) {
        sigma[j] = 0.1 * (bounds[2 * j + 1] - bounds[2 * j]);
    }

    Solution *mutants = (Solution *)malloc(MFA_NUM_MUTANTS * sizeof(Solution));
    for (int i = 0; i < MFA_NUM_MUTANTS; i++) {
        mutants[i].position = (double *)malloc(opt->dim * sizeof(double));
        int idx = xorshift32() % opt->population_size;
        memcpy(mutants[i].position, opt->population[idx].position, opt->dim * sizeof(double));

        for (int m = 0; m < n_mu; m++) {
            int j = xorshift32() % n_var;
            mutants[i].position[j] += sigma[j] * (rand_double_mfa(0.0, 1.0) - 0.5);
            mutants[i].position[j] = fmax(bounds[2 * j], fmin(bounds[2 * j + 1], mutants[i].position[j]));
        }

        mutants[i].fitness = objective_function(mutants[i].position);
        if (mutants[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = mutants[i].fitness;
            memcpy(opt->best_solution.position, mutants[i].position, opt->dim * sizeof(double));
        }
    }

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

    for (int i = 0; i < opt->population_size; i++) {
        free(temp_pop[i].position);
    }
    free(temp_pop);
    for (int i = 0; i < MFA_NUM_MUTANTS; i++) {
        free(mutants[i].position);
    }
    free(mutants);
    free(sigma);
}

// Sort population and select best individuals
void mfa_sort_and_select(Optimizer *opt, Solution *population, int pop_size) {
    // Use qsort instead of bubble sort
    int compare_solutions(const void *a, const void *b) {
        double fa = ((Solution *)a)->fitness;
        double fb = ((Solution *)b)->fitness;
        return (fa > fb) - (fa < fb);
    }
    qsort(population, pop_size, sizeof(Solution), compare_solutions);

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
    xorshift_seed_mfa((unsigned int)time(NULL));

    double *vel_max = (double *)malloc(opt->dim * sizeof(double));
    double *vel_min = (double *)malloc(opt->dim * sizeof(double));
    for (int j = 0; j < opt->dim; j++) {
        vel_max[j] = 0.1 * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        vel_min[j] = -vel_max[j];
    }

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

        inertia_weight *= MFA_INERTIA_DAMP;
        nuptial_dance *= MFA_DANCE_DAMP;
        random_flight *= MFA_FLIGHT_DAMP;

        printf("Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }

    free(vel_max);
    free(vel_min);
}
