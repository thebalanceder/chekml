#include "ABC.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to generate a random double between min and max
double abc_rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Roulette Wheel Selection
int abc_roulette_wheel_selection(double *probabilities, int size) {
    double r = abc_rand_double(0.0, 1.0);
    double *cumsum = (double *)malloc(size * sizeof(double));
    cumsum[0] = probabilities[0];
    for (int i = 1; i < size; i++) {
        cumsum[i] = cumsum[i - 1] + probabilities[i];
    }
    int selected = 0;
    for (int i = 0; i < size; i++) {
        if (r <= cumsum[i]) {
            selected = i;
            break;
        }
    }
    free(cumsum);
    return selected;
}

// Employed Bee Phase
void employed_bee_phase(Optimizer *opt, double (*objective_function)(double *), int *trial_counters) {
    for (int i = 0; i < opt->population_size; i++) {
        // Select random bee (k != i)
        int k;
        do {
            k = rand() % opt->population_size;
        } while (k == i);
        
        // Generate acceleration coefficient
        double *phi = (double *)malloc(opt->dim * sizeof(double));
        for (int j = 0; j < opt->dim; j++) {
            phi[j] = ABC_ACCELERATION_BOUND * abc_rand_double(-1.0, 1.0);
        }
        
        // Create new solution
        double *new_position = (double *)malloc(opt->dim * sizeof(double));
        for (int j = 0; j < opt->dim; j++) {
            new_position[j] = opt->population[i].position[j] + 
                             phi[j] * (opt->population[i].position[j] - opt->population[k].position[j]);
            new_position[j] = fmin(fmax(new_position[j], opt->bounds[2 * j]), opt->bounds[2 * j + 1]);
        }
        
        // Evaluate new solution
        double new_cost = objective_function(new_position);
        
        // Greedy selection
        if (new_cost <= opt->population[i].fitness) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = new_position[j];
            }
            opt->population[i].fitness = new_cost;
            trial_counters[i] = 0;
        } else {
            trial_counters[i]++;
        }
        
        free(phi);
        free(new_position);
    }
}

// Onlooker Bee Phase
void onlooker_bee_phase(Optimizer *opt, double (*objective_function)(double *), int *trial_counters) {
    // Calculate fitness and selection probabilities
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    double mean_cost = 0.0;
    for (int i = 0; i < opt->population_size; i++) {
        mean_cost += opt->population[i].fitness;
    }
    mean_cost /= opt->population_size;
    
    double fitness_sum = 0.0;
    for (int i = 0; i < opt->population_size; i++) {
        fitness[i] = exp(-opt->population[i].fitness / mean_cost);
        fitness_sum += fitness[i];
    }
    
    double *probabilities = (double *)malloc(opt->population_size * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        probabilities[i] = fitness[i] / fitness_sum;
    }
    
    int n_onlookers = (int)(ABC_ONLOOKER_RATIO * opt->population_size);
    for (int m = 0; m < n_onlookers; m++) {
        // Select food source using roulette wheel
        int i = abc_roulette_wheel_selection(probabilities, opt->population_size);
        
        // Select random bee (k != i)
        int k;
        do {
            k = rand() % opt->population_size;
        } while (k == i);
        
        // Generate acceleration coefficient
        double *phi = (double *)malloc(opt->dim * sizeof(double));
        for (int j = 0; j < opt->dim; j++) {
            phi[j] = ABC_ACCELERATION_BOUND * abc_rand_double(-1.0, 1.0);
        }
        
        // Create new solution
        double *new_position = (double *)malloc(opt->dim * sizeof(double));
        for (int j = 0; j < opt->dim; j++) {
            new_position[j] = opt->population[i].position[j] + 
                             phi[j] * (opt->population[i].position[j] - opt->population[k].position[j]);
            new_position[j] = fmin(fmax(new_position[j], opt->bounds[2 * j]), opt->bounds[2 * j + 1]);
        }
        
        // Evaluate new solution
        double new_cost = objective_function(new_position);
        
        // Greedy selection
        if (new_cost <= opt->population[i].fitness) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = new_position[j];
            }
            opt->population[i].fitness = new_cost;
            trial_counters[i] = 0;
        } else {
            trial_counters[i]++;
        }
        
        free(phi);
        free(new_position);
    }
    
    free(fitness);
    free(probabilities);
}

// Scout Bee Phase
void scout_bee_phase(Optimizer *opt, double (*objective_function)(double *), int *trial_counters) {
    int trial_limit = (int)(ABC_TRIAL_LIMIT_FACTOR * opt->dim * opt->population_size);
    for (int i = 0; i < opt->population_size; i++) {
        if (trial_counters[i] >= trial_limit) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = opt->bounds[2 * j] + 
                                                abc_rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
            }
            opt->population[i].fitness = objective_function(opt->population[i].position);
            trial_counters[i] = 0;
        }
    }
}

// Update Best Solution
void abc_update_best_solution(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
}

// Main Optimization Function
void ABC_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize trial counters
    int *trial_counters = (int *)malloc(opt->population_size * sizeof(int));
    for (int i = 0; i < opt->population_size; i++) {
        trial_counters[i] = 0;
    }
    
    // Initialize population
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            abc_rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
    
    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    abc_update_best_solution(opt, objective_function);
    
    // Main loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        employed_bee_phase(opt, objective_function, trial_counters);
        onlooker_bee_phase(opt, objective_function, trial_counters);
        scout_bee_phase(opt, objective_function, trial_counters);
        abc_update_best_solution(opt, objective_function);
        
        printf("Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }
    
    free(trial_counters);
}
