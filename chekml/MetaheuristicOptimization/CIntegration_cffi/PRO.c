#include "PRO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Quicksort implementation with indices
void quicksort_with_indices_pro(double *arr, int *indices, int low, int high) {
    if (low < high) {
        // Partition
        double pivot = arr[indices[high]];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[indices[j]] >= pivot) {  // Descending order
                i++;
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
        int temp = indices[i + 1];
        indices[i + 1] = indices[high];
        indices[high] = temp;
        int pi = i + 1;

        // Recursive calls
        quicksort_with_indices_pro(arr, indices, low, pi - 1);
        quicksort_with_indices_pro(arr, indices, pi + 1, high);
    }
}

// Function to compute the variance of a schedule
double compute_schedule_variance(double *schedule, int dim) {
    double mean = 0.0;
    for (int i = 0; i < dim; i++) {
        mean += schedule[i];
    }
    mean /= dim;
    
    double variance = 0.0;
    for (int i = 0; i < dim; i++) {
        variance += (schedule[i] - mean) * (schedule[i] - mean);
    }
    return variance / dim;
}

// Behavior Selection Phase with partial selection
void select_behaviors(Optimizer *opt, int i, int current_eval, double **schedules, int *selected_behaviors, int *landa) {
    double tau = (double)current_eval / PRO_MAX_EVALUATIONS;
    double selection_rate = exp(-(1.0 - tau));
    
    double *sched = schedules[i];
    int *indices = malloc(opt->dim * sizeof(int));
    for (int j = 0; j < opt->dim; j++) {
        indices[j] = j;
    }
    
    // Use quicksort for descending order
    quicksort_with_indices_pro(sched, indices, 0, opt->dim - 1);
    
    *landa = (int)ceil(opt->dim * rand_double(0.0, 1.0) * selection_rate);
    if (*landa > opt->dim) *landa = opt->dim;
    if (*landa < 1) *landa = 1;
    
    for (int j = 0; j < *landa; j++) {
        selected_behaviors[j] = indices[j];
    }
    
    free(indices);
}

// Behavior Stimulation Phase
void stimulate_behaviors(Optimizer *opt, int i, int *selected_behaviors, int landa, int k, int current_eval, double **schedules, double *new_solution) {
    double tau = (double)current_eval / PRO_MAX_EVALUATIONS;
    
    // Initialize new_solution with current position
    memcpy(new_solution, opt->population[i].position, opt->dim * sizeof(double));
    
    // Apply stimulation
    double sf;
    if (rand_double(0.0, 1.0) < 0.5) {
        // Calculate Stimulation Factor (SF)
        double schedule_mean = 0.0;
        double schedule_max = 0.0;
        for (int j = 0; j < landa; j++) {
            int idx = selected_behaviors[j];
            schedule_mean += schedules[i][idx];
            schedule_max = fmax(schedule_max, fabs(schedules[i][idx]));
        }
        schedule_mean /= landa > 0 ? landa : 1.0;
        sf = tau + rand_double(0.0, 1.0) * (schedule_mean / (schedule_max > 0 ? schedule_max : 1.0));
        
        // Update new solution using best solution
        for (int j = 0; j < landa; j++) {
            int idx = selected_behaviors[j];
            new_solution[idx] += sf * (opt->best_solution.position[idx] - opt->population[i].position[idx]);
        }
    } else {
        // Calculate Stimulation Factor (SF)
        double schedule_mean = 0.0;
        double schedule_max = 0.0;
        for (int j = 0; j < landa; j++) {
            int idx = selected_behaviors[j];
            schedule_mean += schedules[i][idx];
            schedule_max = fmax(schedule_max, fabs(schedules[i][idx]));
        }
        schedule_mean /= landa > 0 ? landa : 1.0;
        sf = tau + rand_double(0.0, 1.0) * (schedule_mean / (schedule_max > 0 ? schedule_max : 1.0));
        
        // Update new solution using another individual
        for (int j = 0; j < landa; j++) {
            int idx = selected_behaviors[j];
            new_solution[idx] += sf * (opt->population[i].position[idx] - opt->population[k].position[idx]);
        }
    }
    
    // Delegate bound constraints to generaloptimizer
    enforce_bound_constraints(opt);
}

// Reinforcement Phase
void apply_reinforcement(Optimizer *opt, int i, int *selected_behaviors, int landa, double **schedules, double *new_solution, double new_fitness) {
    double current_fitness = opt->population[i].fitness;
    
    if (new_fitness < current_fitness) {
        // Positive Reinforcement
        for (int j = 0; j < landa; j++) {
            int idx = selected_behaviors[j];
            schedules[i][idx] += schedules[i][idx] * (REINFORCEMENT_RATE / 2.0);
        }
        // Update population
        memcpy(opt->population[i].position, new_solution, opt->dim * sizeof(double));
        opt->population[i].fitness = new_fitness;
        
        // Update best solution
        if (new_fitness < opt->best_solution.fitness) {
            memcpy(opt->best_solution.position, new_solution, opt->dim * sizeof(double));
            opt->best_solution.fitness = new_fitness;
        }
    } else {
        // Negative Reinforcement
        for (int j = 0; j < landa; j++) {
            int idx = selected_behaviors[j];
            schedules[i][idx] -= schedules[i][idx] * REINFORCEMENT_RATE;
        }
    }
}

// Rescheduling Phase
void reschedule(Optimizer *opt, int i, double **schedules) {
    double variance = compute_schedule_variance(schedules[i], opt->dim);
    if (variance == 0.0) {
        for (int j = 0; j < opt->dim; j++) {
            schedules[i][j] = rand_double(SCHEDULE_MIN, SCHEDULE_MAX);
            opt->population[i].position[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
        printf("Learner %d is Rescheduled\n", i);
    }
}

// Main Optimization Function
void PRO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    int evaluations = 0;
    double *new_solution = malloc(opt->dim * sizeof(double));
    int *selected_behaviors = malloc(opt->dim * sizeof(int));
    int landa;
    
    // Allocate schedules array
    double **schedules = malloc(opt->population_size * sizeof(double *));
    for (int i = 0; i < opt->population_size; i++) {
        schedules[i] = malloc(opt->dim * sizeof(double));
    }
    
    // Allocate temporary arrays for sorting
    double *temp_pos = malloc(opt->dim * sizeof(double));
    double *temp_schedule = malloc(opt->dim * sizeof(double));
    int *pop_indices = malloc(opt->population_size * sizeof(int));
    double *fitness_values = malloc(opt->population_size * sizeof(double));
    
    // Initialize population and schedules
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
            schedules[i][j] = rand_double(SCHEDULE_MIN, SCHEDULE_MAX);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
        evaluations++;
        pop_indices[i] = i;
        fitness_values[i] = opt->population[i].fitness;
    }
    
    // Set initial best solution
    int best_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[best_idx].fitness) {
            best_idx = i;
        }
    }
    memcpy(opt->best_solution.position, opt->population[best_idx].position, opt->dim * sizeof(double));
    opt->best_solution.fitness = opt->population[best_idx].fitness;
    
    while (evaluations < PRO_MAX_EVALUATIONS) {
        // Sort population by fitness using quicksort
        quicksort_with_indices_pro(fitness_values, pop_indices, 0, opt->population_size - 1);
        
        // Apply sorted order to population and schedules
        for (int i = 0; i < opt->population_size; i++) {
            if (pop_indices[i] != i) {
                // Swap contents of position and schedules
                memcpy(temp_pos, opt->population[i].position, opt->dim * sizeof(double));
                memcpy(temp_schedule, schedules[i], opt->dim * sizeof(double));
                double temp_fitness = opt->population[i].fitness;
                
                memcpy(opt->population[i].position, opt->population[pop_indices[i]].position, opt->dim * sizeof(double));
                memcpy(schedules[i], schedules[pop_indices[i]], opt->dim * sizeof(double));
                opt->population[i].fitness = opt->population[pop_indices[i]].fitness;
                
                memcpy(opt->population[pop_indices[i]].position, temp_pos, opt->dim * sizeof(double));
                memcpy(schedules[pop_indices[i]], temp_schedule, opt->dim * sizeof(double));
                opt->population[pop_indices[i]].fitness = temp_fitness;
                
                // Update fitness_values and pop_indices
                fitness_values[i] = opt->population[i].fitness;
                fitness_values[pop_indices[i]] = opt->population[pop_indices[i]].fitness;
                pop_indices[pop_indices[i]] = pop_indices[i];
                pop_indices[i] = i;
            }
        }
        
        for (int i = 0; i < opt->population_size; i++) {
            // Select another individual for comparison
            int k = opt->population_size - 1;
            if (i < opt->population_size - 1) {
                k = i + 1 + (int)(rand_double(0.0, 1.0) * (opt->population_size - i - 1));
            }
            
            // Select behaviors
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
            
            // Update best solution if necessary
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
                opt->best_solution.fitness = opt->population[i].fitness;
            }
            
            // Update fitness_values for next sort
            fitness_values[i] = opt->population[i].fitness;
            
            if (evaluations >= PRO_MAX_EVALUATIONS) {
                break;
            }
        }
        
        printf("Iteration %d: Best Value = %f\n", evaluations, opt->best_solution.fitness);
    }
    
    // Free allocated memory
    free(temp_pos);
    free(temp_schedule);
    free(pop_indices);
    free(fitness_values);
    for (int i = 0; i < opt->population_size; i++) {
        free(schedules[i]);
    }
    free(schedules);
    free(new_solution);
    free(selected_behaviors);
}
