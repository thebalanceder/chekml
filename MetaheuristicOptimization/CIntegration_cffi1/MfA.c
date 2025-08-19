#include "MfA.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Fast random number generator
static unsigned int xorshift_state = 1;

void xorshift_seed_mfa(unsigned int seed) {
    xorshift_state = seed ? seed : (unsigned int)time(NULL);
}

static inline unsigned int xorshift32_mfa(void) {
    unsigned int x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

static inline double rand_double_mfa(double min, double max) {
    return min + (max - min) * (xorshift32_mfa() * (1.0 / 0xffffffffu));
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
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = pos[j];
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Update male population
void mfa_update_males(Optimizer *opt, double (*objective_function)(double *)) {
    int half_pop = opt->population_size / 2;
    
    // Stack-based arrays for velocities and best positions
    double velocities[MFA_POP_SIZE / 2][MFA_DIM_MAX] = {0};
    double best_positions[MFA_POP_SIZE / 2][MFA_DIM_MAX];
    double best_fitnesses[MFA_POP_SIZE / 2];
    double vel_max[MFA_DIM_MAX], vel_min[MFA_DIM_MAX];
    
    // Initialize velocity bounds
    for (int j = 0; j < opt->dim; j++) {
        vel_max[j] = 0.1 * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        vel_min[j] = -vel_max[j];
    }
    
    // Initialize best positions and fitnesses
    for (int i = 0; i < half_pop; i++) {
        for (int j = 0; j < opt->dim; j++) {
            best_positions[i][j] = opt->population[i].position[j];
        }
        best_fitnesses[i] = opt->population[i].fitness;
    }

    for (int i = 0; i < half_pop; i++) {
        double *pos = opt->population[i].position;
        double *vel = velocities[i];
        double *best_pos = best_positions[i];
        
        if (opt->population[i].fitness > opt->best_solution.fitness) {
            for (int j = 0; j < opt->dim; j++) {
                double rpbest = best_pos[j] - pos[j];
                double rgbest = opt->best_solution.position[j] - pos[j];
                vel[j] = (MFA_INERTIA_WEIGHT * vel[j] +
                         MFA_PERSONAL_COEFF * exp(-MFA_DISTANCE_COEFF * rpbest * rpbest) * rpbest +
                         MFA_GLOBAL_COEFF1 * exp(-MFA_DISTANCE_COEFF * rgbest * rgbest) * rgbest);
                vel[j] = fmax(vel_min[j], fmin(vel_max[j], vel[j]));
                pos[j] += vel[j];
            }
        } else {
            for (int j = 0; j < opt->dim; j++) {
                vel[j] = MFA_INERTIA_WEIGHT * vel[j] + MFA_NUPTIAL_DANCE * rand_double_mfa(-1.0, 1.0);
                vel[j] = fmax(vel_min[j], fmin(vel_max[j], vel[j]));
                pos[j] += vel[j];
            }
        }

        opt->population[i].fitness = objective_function(pos);
        
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
    enforce_bound_constraints(opt);
}

// Update female population
void mfa_update_females(Optimizer *opt, double (*objective_function)(double *)) {
    int half_pop = opt->population_size / 2;
    
    double velocities[MFA_POP_SIZE / 2][MFA_DIM_MAX] = {0};
    double vel_max[MFA_DIM_MAX], vel_min[MFA_DIM_MAX];
    
    for (int j = 0; j < opt->dim; j++) {
        vel_max[j] = 0.1 * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        vel_min[j] = -vel_max[j];
    }

    for (int i = 0; i < half_pop; i++) {
        int female_idx = i + half_pop;
        double *male_pos = opt->population[i].position;
        double *female_pos = opt->population[female_idx].position;
        double *vel = velocities[i];
        
        if (opt->population[female_idx].fitness > opt->population[i].fitness) {
            for (int j = 0; j < opt->dim; j++) {
                double rmf = male_pos[j] - female_pos[j];
                vel[j] = (MFA_INERTIA_WEIGHT * vel[j] +
                         MFA_GLOBAL_COEFF2 * exp(-MFA_DISTANCE_COEFF * rmf * rmf) * rmf);
                vel[j] = fmax(vel_min[j], fmin(vel_max[j], vel[j]));
                female_pos[j] += vel[j];
            }
        } else {
            for (int j = 0; j < opt->dim; j++) {
                vel[j] = MFA_INERTIA_WEIGHT * vel[j] + MFA_RANDOM_FLIGHT * rand_double_mfa(-1.0, 1.0);
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
void mfa_mating_phase(Optimizer *opt, double (*objective_function)(double *)) {
    int half_pop = opt->population_size / 2;
    Solution offspring[MFA_NUM_OFFSPRING];
    
    for (int i = 0; i < MFA_NUM_OFFSPRING; i++) {
        offspring[i].position = (double[MFA_DIM_MAX]){0};
    }

    for (int k = 0; k < MFA_NUM_OFFSPRING / 2; k++) {
        double *off1 = offspring[2 * k].position;
        double *off2 = offspring[2 * k + 1].position;
        crossover(opt->population[k].position, opt->population[k + half_pop].position, 
                  off1, off2, opt->dim, opt->bounds);
        
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

    Solution temp_pop[MFA_POP_SIZE];
    for (int i = 0; i < opt->population_size; i++) {
        temp_pop[i].position = (double[MFA_DIM_MAX]){0};
        for (int j = 0; j < opt->dim; j++) {
            temp_pop[i].position[j] = opt->population[i].position[j];
        }
        temp_pop[i].fitness = opt->population[i].fitness;
    }

    for (int i = 0; i < MFA_NUM_OFFSPRING / 2; i++) {
        for (int j = 0; j < opt->dim; j++) {
            temp_pop[half_pop - 1 - i].position[j] = offspring[2 * i].position[j];
            temp_pop[opt->population_size - 1 - i].position[j] = offspring[2 * i + 1].position[j];
        }
        temp_pop[half_pop - 1 - i].fitness = offspring[2 * i].fitness;
        temp_pop[opt->population_size - 1 - i].fitness = offspring[2 * i + 1].fitness;
    }

    mfa_sort_and_select(opt, temp_pop, opt->population_size);
}

// Mutation phase
void mfa_mutation_phase(Optimizer *opt, double (*objective_function)(double *)) {
    int n_var = opt->dim;
    int n_mu = (int)(MFA_MUTATION_RATE * n_var);
    double sigma[MFA_DIM_MAX];
    for (int j = 0; j < n_var; j++) {
        sigma[j] = 0.1 * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
    }

    Solution mutants[MFA_NUM_MUTANTS];
    for (int i = 0; i < MFA_NUM_MUTANTS; i++) {
        mutants[i].position = (double[MFA_DIM_MAX]){0};
        int idx = xorshift32_mfa() % opt->population_size;
        for (int j = 0; j < opt->dim; j++) {
            mutants[i].position[j] = opt->population[idx].position[j];
        }

        for (int m = 0; m < n_mu; m++) {
            int j = xorshift32_mfa() % n_var;
            mutants[i].position[j] += sigma[j] * (rand_double_mfa(0.0, 1.0) - 0.5);
            mutants[i].position[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], mutants[i].position[j]));
        }

        mutants[i].fitness = objective_function(mutants[i].position);
        if (mutants[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = mutants[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = mutants[i].position[j];
            }
        }
    }

    Solution temp_pop[MFA_POP_SIZE];
    for (int i = 0; i < opt->population_size; i++) {
        temp_pop[i].position = (double[MFA_DIM_MAX]){0};
        for (int j = 0; j < opt->dim; j++) {
            temp_pop[i].position[j] = opt->population[i].position[j];
        }
        temp_pop[i].fitness = opt->population[i].fitness;
    }

    for (int i = 0; i < MFA_NUM_MUTANTS; i++) {
        for (int j = 0; j < opt->dim; j++) {
            temp_pop[opt->population_size - 1 - i].position[j] = mutants[i].position[j];
        }
        temp_pop[opt->population_size - 1 - i].fitness = mutants[i].fitness;
    }

    mfa_sort_and_select(opt, temp_pop, opt->population_size);
}

// Sort population and select best individuals
void mfa_sort_and_select(Optimizer *opt, Solution *population, int pop_size) {
    int compare_solutions(const void *a, const void *b) {
        double fa = ((Solution *)a)->fitness;
        double fb = ((Solution *)b)->fitness;
        return (fa > fb) - (fa < fb);
    }
    qsort(population, pop_size, sizeof(Solution), compare_solutions);

    int half_pop = pop_size / 2;
    for (int i = 0; i < half_pop; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = population[i].position[j];
            opt->population[i + half_pop].position[j] = population[half_pop + i].position[j];
        }
        opt->population[i].fitness = population[i].fitness;
        opt->population[i + half_pop].fitness = population[half_pop + i].fitness;
    }
}

// Main optimization function
void MfA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    xorshift_seed_mfa((unsigned int)time(NULL));

    double inertia_weight = MFA_INERTIA_WEIGHT;
    double nuptial_dance = MFA_NUPTIAL_DANCE;
    double random_flight = MFA_RANDOM_FLIGHT;

    mfa_initialize_populations(opt, objective_function);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        mfa_update_males(opt, objective_function);
        mfa_update_females(opt, objective_function);
        mfa_mating_phase(opt, objective_function);
        mfa_mutation_phase(opt, objective_function);

        inertia_weight *= MFA_INERTIA_DAMP;
        nuptial_dance *= MFA_DANCE_DAMP;
        random_flight *= MFA_FLIGHT_DAMP;

        printf("Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
