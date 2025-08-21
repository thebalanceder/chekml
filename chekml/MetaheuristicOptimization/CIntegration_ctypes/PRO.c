#include "PRO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

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

// Behavior Selection Phase
void select_behaviors(Optimizer *opt, int i, int current_eval, double **schedules, int *selected_behaviors, int *landa) {
    double tau = (double)current_eval / PRO_MAX_EVALUATIONS;
    double selection_rate = exp(-(1.0 - tau));
    
    // Sort schedules in descending order
    double *sched = schedules[i];
    int *indices = malloc(opt->dim * sizeof(int));
    for (int j = 0; j < opt->dim; j++) {
        indices[j] = j;
    }
    
    // Simple bubble sort for descending order
    for (int j = 0; j < opt->dim - 1; j++) {
        for (int k = 0; k < opt->dim - j - 1; k++) {
            if (sched[indices[k]] < sched[indices[k + 1]]) {
                int temp = indices[k];
                indices[k] = indices[k + 1];
                indices[k + 1] = temp;
            }
        }
    }
    
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
    double *stimulation = calloc(opt->dim, sizeof(double));
    
    // Copy current position
    for (int j = 0; j < opt->dim; j++) {
        new_solution[j] = opt->population[i].position[j];
    }
    
    // Apply stimulation
    if (rand_double(0.0, 1.0) < 0.5) {
        for (int j = 0; j < landa; j++) {
            int idx = selected_behaviors[j];
            stimulation[idx] = opt->best_solution.position[idx] - opt->population[i].position[idx];
        }
    } else {
        for (int j = 0; j < landa; j++) {
            int idx = selected_behaviors[j];
            stimulation[idx] = opt->population[i].position[idx] - opt->population[k].position[idx];
        }
    }
    
    // Calculate Stimulation Factor (SF)
    double schedule_mean = 0.0;
    double schedule_max = 0.0;
    for (int j = 0; j < landa; j++) {
        int idx = selected_behaviors[j];
        schedule_mean += schedules[i][idx];
        schedule_max = fmax(schedule_max, fabs(schedules[i][idx]));
    }
    schedule_mean /= landa > 0 ? landa : 1.0;
    double sf = tau + rand_double(0.0, 1.0) * (schedule_mean / (schedule_max > 0 ? schedule_max : 1.0));
    
    // Update new solution
    for (int j = 0; j < landa; j++) {
        int idx = selected_behaviors[j];
        new_solution[idx] += sf * stimulation[idx];
    }
    
    // Bound constraints control
    for (int j = 0; j < opt->dim; j++) {
        if (new_solution[j] < opt->bounds[2 * j]) {
            new_solution[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        if (new_solution[j] > opt->bounds[2 * j + 1]) {
            new_solution[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
    }
    
    free(stimulation);
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
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = new_solution[j];
        }
        opt->population[i].fitness = new_fitness;
        
        // Update best solution
        if (new_fitness < opt->best_solution.fitness) {
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = new_solution[j];
            }
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
    
    // Initialize population and schedules
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
            schedules[i][j] = rand_double(SCHEDULE_MIN, SCHEDULE_MAX);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
        evaluations++;
    }
    
    // Set initial best solution
    int best_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[best_idx].fitness) {
            best_idx = i;
        }
    }
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[best_idx].position[j];
    }
    opt->best_solution.fitness = opt->population[best_idx].fitness;
    
    while (evaluations < PRO_MAX_EVALUATIONS) {
        // Sort population by fitness
        for (int i = 0; i < opt->population_size - 1; i++) {
            for (int j = 0; j < opt->population_size - i - 1; j++) {
                if (opt->population[j].fitness > opt->population[j + 1].fitness) {
                    // Swap contents of position and schedules arrays
                    double *temp_pos = malloc(opt->dim * sizeof(double));
                    double *temp_schedule = malloc(opt->dim * sizeof(double));
                    double temp_fitness;
                    
                    // Copy position[j] to temp_pos
                    memcpy(temp_pos, opt->population[j].position, opt->dim * sizeof(double));
                    // Copy position[j+1] to position[j]
                    memcpy(opt->population[j].position, opt->population[j + 1].position, opt->dim * sizeof(double));
                    // Copy temp_pos to position[j+1]
                    memcpy(opt->population[j + 1].position, temp_pos, opt->dim * sizeof(double));
                    
                    // Copy schedules[j] to temp_schedule
                    memcpy(temp_schedule, schedules[j], opt->dim * sizeof(double));
                    // Copy schedules[j+1] to schedules[j]
                    memcpy(schedules[j], schedules[j + 1], opt->dim * sizeof(double));
                    // Copy temp_schedule to schedules[j+1]
                    memcpy(schedules[j + 1], temp_schedule, opt->dim * sizeof(double));
                    
                    // Swap fitness
                    temp_fitness = opt->population[j].fitness;
                    opt->population[j].fitness = opt->population[j + 1].fitness;
                    opt->population[j + 1].fitness = temp_fitness;
                    
                    free(temp_pos);
                    free(temp_schedule);
                }
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
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
                opt->best_solution.fitness = opt->population[i].fitness;
            }
            
            if (evaluations >= PRO_MAX_EVALUATIONS) {
                break;
            }
        }
        
        printf("Iteration %d: Best Value = %f\n", evaluations, opt->best_solution.fitness);
    }
    
    // Free schedules array
    for (int i = 0; i < opt->population_size; i++) {
        free(schedules[i]);
    }
    free(schedules);
    free(new_solution);
    free(selected_behaviors);
}
