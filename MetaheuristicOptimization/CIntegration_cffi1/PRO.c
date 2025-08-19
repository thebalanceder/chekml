#include "PRO.h"
#include "generaloptimizer.h"
#include <string.h>

double rand_double(double min, double max);

// Top-k selection for descending order
void select_top_k(double *arr, int *indices, int n, int k) {
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }
    for (int i = 0; i < k; i++) {
        int max_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[indices[j]] > arr[indices[max_idx]]) {
                max_idx = j;
            }
        }
        int temp = indices[i];
        indices[i] = indices[max_idx];
        indices[max_idx] = temp;
    }
}

// Compute schedule variance inline
static inline double compute_schedule_variance(double *schedule, int dim) {
    double mean = 0.0;
    for (int i = 0; i < dim; i++) {
        mean += schedule[i];
    }
    mean /= dim;
    
    double variance = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = schedule[i] - mean;
        variance += diff * diff;
    }
    return variance / dim;
}

// Behavior Selection Phase
void select_behaviors(Optimizer *opt, int i, int current_eval, double **schedules, int *selected_behaviors, int *landa) {
    const int dim = opt->dim;
    const double tau = (double)current_eval / PRO_MAX_EVALUATIONS;
    const double selection_rate = exp(-(1.0 - tau));
    
    *landa = (int)(dim * rand_double(0.0, 1.0) * selection_rate + 0.5);
    if (*landa > dim) *landa = dim;
    if (*landa < 1) *landa = 1;
    
    // Use top-k selection for schedules
    select_top_k(schedules[i], selected_behaviors, dim, *landa);
}

// Behavior Stimulation Phase
void stimulate_behaviors(Optimizer *opt, int i, int *selected_behaviors, int landa, int k, int current_eval, double **schedules, double *new_solution) {
    const int dim = opt->dim;
    const double tau = (double)current_eval / PRO_MAX_EVALUATIONS;
    
    // Copy current position
    memcpy(new_solution, opt->population[i].position, dim * sizeof(double));
    
    // Calculate Stimulation Factor (SF)
    double schedule_mean = 0.0;
    double schedule_max = 0.0;
    for (int j = 0; j < landa; j++) {
        int idx = selected_behaviors[j];
        double sched_val = schedules[i][idx];
        schedule_mean += sched_val;
        schedule_max = fmax(schedule_max, fabs(sched_val));
    }
    schedule_mean /= landa > 0 ? landa : 1.0;
    const double sf = tau + rand_double(0.0, 1.0) * (schedule_mean / (schedule_max > 0 ? schedule_max : 1.0));
    
    // Apply stimulation
    if (rand_double(0.0, 1.0) < 0.5) {
        for (int j = 0; j < landa; j++) {
            int idx = selected_behaviors[j];
            new_solution[idx] += sf * (opt->best_solution.position[idx] - opt->population[i].position[idx]);
        }
    } else {
        for (int j = 0; j < landa; j++) {
            int idx = selected_behaviors[j];
            new_solution[idx] += sf * (opt->population[i].position[idx] - opt->population[k].position[idx]);
        }
    }
    
    // Delegate bound constraints
    enforce_bound_constraints(opt);
}

// Reinforcement Phase
void apply_reinforcement(Optimizer *opt, int i, int *selected_behaviors, int landa, double **schedules, double *new_solution, double new_fitness) {
    const double current_fitness = opt->population[i].fitness;
    const int dim = opt->dim;
    
    if (new_fitness < current_fitness) {
        // Positive Reinforcement
        for (int j = 0; j < landa; j++) {
            int idx = selected_behaviors[j];
            schedules[i][idx] *= (1.0 + REINFORCEMENT_RATE / 2.0);
        }
        // Update population
        memcpy(opt->population[i].position, new_solution, dim * sizeof(double));
        opt->population[i].fitness = new_fitness;
        
        // Update best solution
        if (new_fitness < opt->best_solution.fitness) {
            memcpy(opt->best_solution.position, new_solution, dim * sizeof(double));
            opt->best_solution.fitness = new_fitness;
        }
    } else {
        // Negative Reinforcement
        for (int j = 0; j < landa; j++) {
            int idx = selected_behaviors[j];
            schedules[i][idx] *= (1.0 - REINFORCEMENT_RATE);
        }
    }
}

// Rescheduling Phase
void reschedule(Optimizer *opt, int i, double **schedules) {
    const int dim = opt->dim;
    if (compute_schedule_variance(schedules[i], dim) == 0.0) {
        for (int j = 0; j < dim; j++) {
            schedules[i][j] = rand_double(SCHEDULE_MIN, SCHEDULE_MAX);
            opt->population[i].position[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
}

// Main Optimization Function
void PRO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    const int pop_size = opt->population_size;
    const int dim = opt->dim;
    int evaluations = 0;
    
    // Pre-allocate all arrays
    double *new_solution = malloc(dim * sizeof(double));
    int *selected_behaviors = malloc(dim * sizeof(int));
    double **schedules = malloc(pop_size * sizeof(double *));
    double *schedules_data = malloc(pop_size * dim * sizeof(double));
    double *temp_pos = malloc(dim * sizeof(double));
    int *pop_indices = malloc(pop_size * sizeof(int));
    double *fitness_values = malloc(pop_size * sizeof(double));
    
    // Initialize schedules in contiguous memory
    for (int i = 0; i < pop_size; i++) {
        schedules[i] = schedules_data + i * dim;
    }
    
    // Initialize population, schedules, and fitness_values
    for (int i = 0; i < pop_size; i++) {
        for (int j = 0; j < dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
            schedules[i][j] = rand_double(SCHEDULE_MIN, SCHEDULE_MAX);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
        fitness_values[i] = opt->population[i].fitness;
        evaluations++;
        pop_indices[i] = i;
    }
    
    // Set initial best solution
    int best_idx = 0;
    for (int i = 1; i < pop_size; i++) {
        if (opt->population[i].fitness < opt->population[best_idx].fitness) {
            best_idx = i;
        }
    }
    memcpy(opt->best_solution.position, opt->population[best_idx].position, dim * sizeof(double));
    opt->best_solution.fitness = opt->population[best_idx].fitness;
    
    // Select top 10% of population for updates
    const int top_k = pop_size / 10 > 0 ? pop_size / 10 : 1;
    
    while (evaluations < PRO_MAX_EVALUATIONS) {
        // Select top-k individuals
        select_top_k(fitness_values, pop_indices, pop_size, top_k);
        
        // Process only top-k individuals
        for (int ii = 0; ii < top_k; ii++) {
            int i = pop_indices[ii];
            
            // Select another individual for comparison
            int k = pop_size - 1;
            if (i < pop_size - 1) {
                k = i + 1 + (int)(rand_double(0.0, 1.0) * (pop_size - i - 1));
            }
            
            // Select behaviors
            int landa;
            select_behaviors(opt, i, evaluations, schedules, selected_behaviors, &landa);
            
            // Stimulate behaviors
            stimulate_behaviors(opt, i, selected_behaviors, landa, k, evaluations, schedules, new_solution);
            
            // Evaluate new solution
            double new_fitness = objective_function(new_solution);
            evaluations++;
            
            // Apply reinforcement
            apply_reinforcement(opt, i, selected_behaviors, landa, schedules, new_solution, new_fitness);
            
            // Reschedule if necessary
            reschedule(opt, i, schedules);
            
            // Update best solution
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                memcpy(opt->best_solution.position, opt->population[i].position, dim * sizeof(double));
                opt->best_solution.fitness = opt->population[i].fitness;
            }
            
            // Update fitness_values
            fitness_values[i] = opt->population[i].fitness;
            
            if (evaluations >= PRO_MAX_EVALUATIONS) {
                break;
            }
        }
    }
    
    // Free allocated memory
    free(fitness_values);
    free(temp_pos);
    free(pop_indices);
    free(schedules_data);
    free(schedules);
    free(new_solution);
    free(selected_behaviors);
}
