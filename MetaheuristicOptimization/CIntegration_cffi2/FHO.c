#include "FHO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Generate a random double between min and max
double rand_double(double min, double max);

// Initialize the population randomly within bounds
void initialize_population_fho(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Compute Euclidean distance between two points
double euclidean_distance(double *a, double *b, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Assign prey to firehawks based on distance
void assign_prey_to_firehawks(Optimizer *opt, int num_firehawks, int *prey_counts, int **prey_indices) {
    int prey_size = opt->population_size - num_firehawks;
    if (prey_size <= 0) {
        for (int i = 0; i < num_firehawks; i++) {
            prey_counts[i] = 0;
        }
        return;
    }

    double *distances = (double *)malloc(prey_size * sizeof(double));
    int *sorted_indices = (int *)malloc(prey_size * sizeof(int));
    int *remaining_prey = (int *)malloc(prey_size * sizeof(int));
    int remaining_count = prey_size;

    // Initialize remaining prey indices
    for (int i = 0; i < prey_size; i++) {
        remaining_prey[i] = num_firehawks + i;
    }

    for (int i = 0; i < num_firehawks; i++) {
        if (remaining_count == 0) {
            prey_counts[i] = 0;
            continue;
        }

        // Calculate distances from firehawk i to remaining prey
        for (int j = 0; j < remaining_count; j++) {
            distances[j] = euclidean_distance(opt->population[i].position, 
                                             opt->population[remaining_prey[j]].position, 
                                             opt->dim);
            sorted_indices[j] = j;
        }

        // Sort distances
        for (int j = 0; j < remaining_count - 1; j++) {
            for (int k = j + 1; k < remaining_count; k++) {
                if (distances[sorted_indices[j]] > distances[sorted_indices[k]]) {
                    int temp = sorted_indices[j];
                    sorted_indices[j] = sorted_indices[k];
                    sorted_indices[k] = temp;
                }
            }
        }

        // Randomly select number of prey
        int num_prey = rand() % remaining_count + 1;
        prey_counts[i] = num_prey;
        prey_indices[i] = (int *)malloc(num_prey * sizeof(int));

        // Assign nearest prey
        for (int j = 0; j < num_prey; j++) {
            prey_indices[i][j] = remaining_prey[sorted_indices[j]];
        }

        // Remove assigned prey
        int new_remaining_count = 0;
        for (int j = 0; j < remaining_count; j++) {
            int is_assigned = 0;
            for (int k = 0; k < num_prey; k++) {
                if (j == sorted_indices[k]) {
                    is_assigned = 1;
                    break;
                }
            }
            if (!is_assigned) {
                remaining_prey[new_remaining_count++] = remaining_prey[j];
            }
        }
        remaining_count = new_remaining_count;
    }

    // Assign remaining prey to the last group
    if (remaining_count > 0 && num_firehawks > 0) {
        int last_idx = num_firehawks - 1;
        int new_count = prey_counts[last_idx] + remaining_count;
        int *new_indices = (int *)malloc(new_count * sizeof(int));
        for (int i = 0; i < prey_counts[last_idx]; i++) {
            new_indices[i] = prey_indices[last_idx][i];
        }
        for (int i = 0; i < remaining_count; i++) {
            new_indices[prey_counts[last_idx] + i] = remaining_prey[i];
        }
        free(prey_indices[last_idx]);
        prey_indices[last_idx] = new_indices;
        prey_counts[last_idx] = new_count;
    }

    free(distances);
    free(sorted_indices);
    free(remaining_prey);
}

// Update firehawk position
void update_firehawk_position(Optimizer *opt, double *fh, double *other_fh, double *new_pos) {
    double ir1 = rand_double(IR_MIN, IR_MAX);
    double ir2 = rand_double(IR_MIN, IR_MAX);
    for (int i = 0; i < opt->dim; i++) {
        new_pos[i] = fh[i] + (ir1 * opt->best_solution.position[i] - ir2 * other_fh[i]);
        new_pos[i] = fmax(opt->bounds[2 * i], fmin(opt->bounds[2 * i + 1], new_pos[i]));
    }
}

// Update prey position with two strategies
void update_prey_position(Optimizer *opt, double *prey, double *firehawk, double *local_safe_point, 
                         double *global_safe_point, double *pos1, double *pos2) {
    double ir1 = rand_double(IR_MIN, IR_MAX);
    double ir2 = rand_double(IR_MIN, IR_MAX);
    for (int i = 0; i < opt->dim; i++) {
        pos1[i] = prey[i] + (ir1 * firehawk[i] - ir2 * local_safe_point[i]);
        pos1[i] = fmax(opt->bounds[2 * i], fmin(opt->bounds[2 * i + 1], pos1[i]));
    }

    ir1 = rand_double(IR_MIN, IR_MAX);
    ir2 = rand_double(IR_MIN, IR_MAX);
    int rand_fh_idx = rand() % opt->population_size;  // Random firehawk for pos2
    double *other_fh = opt->population[rand_fh_idx].position;
    for (int i = 0; i < opt->dim; i++) {
        pos2[i] = prey[i] + (ir1 * other_fh[i] - ir2 * global_safe_point[i]);
        pos2[i] = fmax(opt->bounds[2 * i], fmin(opt->bounds[2 * i + 1], pos2[i]));
    }
}

// Main optimization function
void FHO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand((unsigned)time(NULL));
    initialize_population_fho(opt);

    // Evaluate initial population
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    double *global_safe_point = (double *)malloc(opt->dim * sizeof(double));
    int iter = 0;
    int fes = opt->population_size;

    while (fes < opt->max_iter) {
        iter++;
        // Compute global safe point (mean of population)
        memset(global_safe_point, 0, opt->dim * sizeof(double));
        for (int i = 0; i < opt->population_size; i++) {
            for (int j = 0; j < opt->dim; j++) {
                global_safe_point[j] += opt->population[i].position[j];
            }
        }
        for (int j = 0; j < opt->dim; j++) {
            global_safe_point[j] /= opt->population_size;
        }

        // Determine number of firehawks
        int max_firehawks = (int)(ceil(opt->population_size * MAX_FIREHAWKS_RATIO)) + 1;
        int num_firehawks = MIN_FIREHAWKS + (rand() % (max_firehawks - MIN_FIREHAWKS + 1));

        // Assign prey to firehawks
        int *prey_counts = (int *)calloc(num_firehawks, sizeof(int));
        int **prey_indices = (int **)malloc(num_firehawks * sizeof(int *));
        assign_prey_to_firehawks(opt, num_firehawks, prey_counts, prey_indices);

        // Temporary population for new solutions
        int max_new_pop_size = opt->population_size + 2 * (opt->population_size - num_firehawks);
        double **new_positions = (double **)malloc(max_new_pop_size * sizeof(double *));
        double *new_fitness = (double *)malloc(max_new_pop_size * sizeof(double));
        int new_pop_size = 0;

        // Update firehawks
        for (int i = 0; i < num_firehawks; i++) {
            int other_fh_idx = rand() % num_firehawks;
            double *new_pos = (double *)malloc(opt->dim * sizeof(double));
            update_firehawk_position(opt, opt->population[i].position, 
                                    opt->population[other_fh_idx].position, new_pos);
            new_positions[new_pop_size] = new_pos;
            new_fitness[new_pop_size] = objective_function(new_pos);
            new_pop_size++;
            fes++;
        }

        // Update prey
        for (int i = 0; i < num_firehawks; i++) {
            if (prey_counts[i] == 0) {
                continue;
            }
            // Compute local safe point
            double *local_safe_point = (double *)malloc(opt->dim * sizeof(double));
            memset(local_safe_point, 0, opt->dim * sizeof(double));
            for (int j = 0; j < prey_counts[i]; j++) {
                for (int k = 0; k < opt->dim; k++) {
                    local_safe_point[k] += opt->population[prey_indices[i][j]].position[k];
                }
            }
            for (int k = 0; k < opt->dim; k++) {
                local_safe_point[k] /= prey_counts[i];
            }

            // Update each prey
            for (int j = 0; j < prey_counts[i]; j++) {
                double *pos1 = (double *)malloc(opt->dim * sizeof(double));
                double *pos2 = (double *)malloc(opt->dim * sizeof(double));
                update_prey_position(opt, opt->population[prey_indices[i][j]].position,
                                    opt->population[i].position, local_safe_point,
                                    global_safe_point, pos1, pos2);
                new_positions[new_pop_size] = pos1;
                new_fitness[new_pop_size] = objective_function(pos1);
                new_pop_size++;
                fes++;
                new_positions[new_pop_size] = pos2;
                new_fitness[new_pop_size] = objective_function(pos2);
                new_pop_size++;
                fes++;
            }
            free(local_safe_point);
        }

        // Sort new population by fitness
        for (int i = 0; i < new_pop_size - 1; i++) {
            for (int j = i + 1; j < new_pop_size; j++) {
                if (new_fitness[i] > new_fitness[j]) {
                    double temp_fitness = new_fitness[i];
                    new_fitness[i] = new_fitness[j];
                    new_fitness[j] = temp_fitness;
                    double *temp_pos = new_positions[i];
                    new_positions[i] = new_positions[j];
                    new_positions[j] = temp_pos;
                }
            }
        }

        // Update main population
        for (int i = 0; i < opt->population_size && i < new_pop_size; i++) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = new_positions[i][j];
            }
            opt->population[i].fitness = new_fitness[i];
            if (new_fitness[i] < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness[i];
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = new_positions[i][j];
                }
            }
        }

        // Clean up
        for (int i = 0; i < new_pop_size; i++) {
            free(new_positions[i]);
        }
        for (int i = 0; i < num_firehawks; i++) {
            if (prey_counts[i] > 0) {
                free(prey_indices[i]);
            }
        }
        free(new_positions);
        free(new_fitness);
        free(prey_counts);
        free(prey_indices);

        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Value = %f\n", iter, opt->best_solution.fitness);
    }

    free(global_safe_point);
}
