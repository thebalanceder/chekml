#include "CO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// 🌊 Generate a random double between min and max
double rand_double(double min, double max);

// 🐦 Initialize Cuckoo Population
void initialize_cuckoos(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;  // Will be updated later
    }
    enforce_bound_constraints(opt);
}

// 🥚 Lay Eggs for Each Cuckoo
void lay_eggs(Optimizer *opt, double **egg_positions, int *num_eggs, int *total_eggs) {
    *total_eggs = 0;
    for (int i = 0; i < opt->population_size; i++) {
        num_eggs[i] = MIN_EGGS + (int)(rand_double(0, 1) * (MAX_EGGS - MIN_EGGS + 1));
        *total_eggs += num_eggs[i];
    }

    // Allocate memory for egg positions
    *egg_positions = (double *)malloc(*total_eggs * opt->dim * sizeof(double));
    if (*egg_positions == NULL) {
        fprintf(stderr, "Memory allocation failed for egg_positions\n");
        exit(1);
    }

    int egg_idx = 0;
    for (int i = 0; i < opt->population_size; i++) {
        double radius_factor = ((double)num_eggs[i] / *total_eggs) * RADIUS_COEFF;
        for (int k = 0; k < num_eggs[i]; k++) {
            double random_scalar = rand_double(0, 1);
            double angle = k * (2 * PI_CO / num_eggs[i]);
            int sign = (rand() % 2 == 0) ? 1 : -1;

            for (int j = 0; j < opt->dim; j++) {
                double bound_range = opt->bounds[2 * j + 1] - opt->bounds[2 * j];
                double radius = radius_factor * random_scalar * bound_range;
                double adding_value = sign * radius * cos(angle) + radius * sin(angle);
                double new_pos = opt->population[i].position[j] + adding_value;

                // Enforce bounds
                if (new_pos < opt->bounds[2 * j]) {
                    new_pos = opt->bounds[2 * j];
                } else if (new_pos > opt->bounds[2 * j + 1]) {
                    new_pos = opt->bounds[2 * j + 1];
                }
                (*egg_positions)[egg_idx * opt->dim + j] = new_pos;
            }
            egg_idx++;
        }
    }
}

// 🏆 Select Best Cuckoos
void select_best_cuckoos(Optimizer *opt, double **positions, double *fitness, int num_positions) {
    if (num_positions <= MAX_CUCKOOS) {
        for (int i = 0; i < num_positions; i++) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = (*positions)[i * opt->dim + j];
            }
            opt->population[i].fitness = fitness[i];
        }
        opt->population_size = num_positions;
        return;
    }

    // Find indices of top MAX_CUCKOOS fitness values
    int *indices = (int *)malloc(num_positions * sizeof(int));
    for (int i = 0; i < num_positions; i++) {
        indices[i] = i;
    }

    // Simple bubble sort for top MAX_CUCKOOS (sufficient for small populations)
    for (int i = 0; i < MAX_CUCKOOS; i++) {
        for (int j = i + 1; j < num_positions; j++) {
            if (fitness[indices[i]] > fitness[indices[j]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }

    // Update population with best cuckoos
    for (int i = 0; i < MAX_CUCKOOS; i++) {
        int idx = indices[i];
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = (*positions)[idx * opt->dim + j];
        }
        opt->population[i].fitness = fitness[idx];
    }
    opt->population_size = MAX_CUCKOOS;

    free(indices);
}

// 🌐 Cluster and Migrate (Simplified without KMeans)
int cluster_and_migrate(Optimizer *opt) {
    // Check variance threshold
    double var_sum = 0.0;
    for (int j = 0; j < opt->dim; j++) {
        double mean = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            mean += opt->population[i].position[j];
        }
        mean /= opt->population_size;
        double variance = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            variance += (opt->population[i].position[j] - mean) * (opt->population[i].position[j] - mean);
        }
        variance /= opt->population_size;
        var_sum += variance;
    }
    if (var_sum < VARIANCE_THRESHOLD) {
        return 1;  // Stop optimization
    }

    // Find best cuckoo (simplified clustering: use best solution as goal)
    int best_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[best_idx].fitness) {
            best_idx = i;
        }
    }

    // Migrate toward best cuckoo
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double new_pos = opt->population[i].position[j] + MOTION_COEFF * rand_double(0, 1) * 
                             (opt->population[best_idx].position[j] - opt->population[i].position[j]);
            if (new_pos < opt->bounds[2 * j]) {
                new_pos = opt->bounds[2 * j];
            } else if (new_pos > opt->bounds[2 * j + 1]) {
                new_pos = opt->bounds[2 * j + 1];
            }
            opt->population[i].position[j] = new_pos;
        }
    }
    enforce_bound_constraints(opt);
    return 0;  // Continue optimization
}

// 🚀 Main Optimization Function
void CO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    initialize_cuckoos(opt);

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

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Lay eggs
        double *egg_positions = NULL;
        int *num_eggs = (int *)malloc(opt->population_size * sizeof(int));
        int total_eggs = 0;
        lay_eggs(opt, &egg_positions, num_eggs, &total_eggs);

        // Evaluate eggs
        double *egg_fitness = (double *)malloc(total_eggs * sizeof(double));
        for (int i = 0; i < total_eggs; i++) {
            egg_fitness[i] = objective_function(&egg_positions[i * opt->dim]);
        }

        // Combine cuckoos and eggs
        int total_positions = opt->population_size + total_eggs;
        double *all_positions = (double *)malloc(total_positions * opt->dim * sizeof(double));
        double *all_fitness = (double *)malloc(total_positions * sizeof(double));

        for (int i = 0; i < opt->population_size; i++) {
            for (int j = 0; j < opt->dim; j++) {
                all_positions[i * opt->dim + j] = opt->population[i].position[j];
            }
            all_fitness[i] = opt->population[i].fitness;
        }
        for (int i = 0; i < total_eggs; i++) {
            for (int j = 0; j < opt->dim; j++) {
                all_positions[(opt->population_size + i) * opt->dim + j] = egg_positions[i * opt->dim + j];
            }
            all_fitness[opt->population_size + i] = egg_fitness[i];
        }

        // Select best cuckoos
        select_best_cuckoos(opt, &all_positions, all_fitness, total_positions);

        // Cluster and migrate
        int stop = cluster_and_migrate(opt);

        // Update best solution
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        // Replace worst cuckoo with best solution if needed
        int worst_idx = 0;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness > opt->population[worst_idx].fitness) {
                worst_idx = i;
            }
        }
        if (opt->population[worst_idx].fitness > opt->best_solution.fitness) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[worst_idx].position[j] = opt->best_solution.position[j];
            }
            opt->population[worst_idx].fitness = opt->best_solution.fitness;
        }

        // Replace second-worst with randomized best (if population > 1)
        if (opt->population_size > 1) {
            int second_worst_idx = (worst_idx == 0) ? 1 : 0;
            for (int i = 0; i < opt->population_size; i++) {
                if (i != worst_idx && opt->population[i].fitness > opt->population[second_worst_idx].fitness) {
                    second_worst_idx = i;
                }
            }
            for (int j = 0; j < opt->dim; j++) {
                double new_pos = opt->best_solution.position[j] * rand_double(0, 1);
                if (new_pos < opt->bounds[2 * j]) {
                    new_pos = opt->bounds[2 * j];
                } else if (new_pos > opt->bounds[2 * j + 1]) {
                    new_pos = opt->bounds[2 * j + 1];
                }
                opt->population[second_worst_idx].position[j] = new_pos;
            }
            opt->population[second_worst_idx].fitness = objective_function(opt->population[second_worst_idx].position);
        }

        // Clean up
        free(egg_positions);
        free(num_eggs);
        free(egg_fitness);
        free(all_positions);
        free(all_fitness);

        if (stop) {
            break;
        }

        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
