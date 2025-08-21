#include "CSO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Function to generate a random integer excluding a tabu value
int randi_tabu(int min_val, int max_val, int tabu) {
    int temp;
    do {
        temp = min_val + (int)((max_val - min_val + 1) * rand_double(0.0, 1.0));
    } while (temp == tabu);
    return temp;
}

// Function to generate a permutation (simplified randperm_f)
void randperm_f(int range_size, int dim, int *result) {
    int *temp = (int *)malloc(range_size * sizeof(int));
    for (int i = 0; i < range_size; i++) {
        temp[i] = i + 1;
    }
    // Shuffle temp
    for (int i = range_size - 1; i > 0; i--) {
        int j = (int)(rand_double(0.0, 1.0) * (i + 1));
        int swap = temp[i];
        temp[i] = temp[j];
        temp[j] = swap;
    }
    // Copy to result, extending if dim > range_size
    for (int i = 0; i < dim; i++) {
        if (i < range_size) {
            result[i] = temp[i];
        } else {
            result[i] = 1 + (int)(rand_double(0.0, 1.0) * range_size);
        }
    }
    free(temp);
}

// Initialize population
void initialize_population_cso(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
    enforce_bound_constraints(opt);

    // Find initial best solution
    int best_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[best_idx].fitness) {
            best_idx = i;
        }
    }
    opt->best_solution.fitness = opt->population[best_idx].fitness;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[best_idx].position[j];
    }
}

// Update roosters
void update_roosters(Optimizer *opt) {
    int rooster_num = (int)(opt->population_size * ROOSTER_RATIO);
    int *sort_indices = (int *)malloc(opt->population_size * sizeof(int));
    // Sort indices by fitness
    for (int i = 0; i < opt->population_size; i++) {
        sort_indices[i] = i;
    }
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = i + 1; j < opt->population_size; j++) {
            if (opt->population[sort_indices[j]].fitness < opt->population[sort_indices[i]].fitness) {
                int temp = sort_indices[i];
                sort_indices[i] = sort_indices[j];
                sort_indices[j] = temp;
            }
        }
    }

    for (int i = 0; i < rooster_num; i++) {
        int curr_idx = sort_indices[i];
        int another_rooster = randi_tabu(1, rooster_num, i + 1);
        int another_idx = sort_indices[another_rooster - 1];

        double sigma;
        if (opt->population[curr_idx].fitness <= opt->population[another_idx].fitness) {
            sigma = 1.0;
        } else {
            sigma = exp((opt->population[another_idx].fitness - opt->population[curr_idx].fitness) /
                        (fabs(opt->population[curr_idx].fitness) + 2.2e-16));
        }

        for (int j = 0; j < opt->dim; j++) {
            opt->population[curr_idx].position[j] *= (1.0 + sigma * rand_double(-3.0, 3.0)); // Approximate normal distribution
        }
    }
    enforce_bound_constraints(opt);
    free(sort_indices);
}

// Update hens
void update_hens(Optimizer *opt) {
    int rooster_num = (int)(opt->population_size * ROOSTER_RATIO);
    int hen_num = (int)(opt->population_size * HEN_RATIO);
    int *sort_indices = (int *)malloc(opt->population_size * sizeof(int));
    for (int i = 0; i < opt->population_size; i++) {
        sort_indices[i] = i;
    }
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = i + 1; j < opt->population_size; j++) {
            if (opt->population[sort_indices[j]].fitness < opt->population[sort_indices[i]].fitness) {
                int temp = sort_indices[i];
                sort_indices[i] = sort_indices[j];
                sort_indices[j] = temp;
            }
        }
    }

    int *mate = (int *)malloc(hen_num * sizeof(int));
    randperm_f(rooster_num, hen_num, mate);

    for (int i = rooster_num; i < rooster_num + hen_num; i++) {
        int curr_idx = sort_indices[i];
        int mate_idx = sort_indices[mate[i - rooster_num] - 1];
        int other = randi_tabu(1, i, mate[i - rooster_num]);

        double c1 = exp((opt->population[curr_idx].fitness - opt->population[mate_idx].fitness) /
                        (fabs(opt->population[curr_idx].fitness) + 2.2e-16));
        double c2 = exp(-opt->population[curr_idx].fitness + opt->population[sort_indices[other - 1]].fitness);

        for (int j = 0; j < opt->dim; j++) {
            opt->population[curr_idx].position[j] += 
                c1 * rand_double(0.0, 1.0) * (opt->population[mate_idx].position[j] - opt->population[curr_idx].position[j]) +
                c2 * rand_double(0.0, 1.0) * (opt->population[sort_indices[other - 1]].position[j] - opt->population[curr_idx].position[j]);
        }
    }
    enforce_bound_constraints(opt);
    free(sort_indices);
    free(mate);
}

// Update chicks
void update_chicks(Optimizer *opt) {
    int rooster_num = (int)(opt->population_size * ROOSTER_RATIO);
    int hen_num = (int)(opt->population_size * HEN_RATIO);
    int chick_num = opt->population_size - rooster_num - hen_num;
    int mother_num = (int)(hen_num * MOTHER_RATIO);

    int *sort_indices = (int *)malloc(opt->population_size * sizeof(int));
    for (int i = 0; i < opt->population_size; i++) {
        sort_indices[i] = i;
    }
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = i + 1; j < opt->population_size; j++) {
            if (opt->population[sort_indices[j]].fitness < opt->population[sort_indices[i]].fitness) {
                int temp = sort_indices[i];
                sort_indices[i] = sort_indices[j];
                sort_indices[j] = temp;
            }
        }
    }

    int *mother_lib = (int *)malloc(hen_num * sizeof(int));
    randperm_f(hen_num, mother_num, mother_lib);
    for (int i = 0; i < mother_num; i++) {
        mother_lib[i] += rooster_num;
    }

    for (int i = rooster_num + hen_num; i < opt->population_size; i++) {
        int curr_idx = sort_indices[i];
        int mother_idx = sort_indices[mother_lib[(int)(rand_double(0.0, 1.0) * mother_num)] - 1];
        double fl = 0.5 + 0.4 * rand_double(0.0, 1.0); // FL in [0.5, 0.9]

        for (int j = 0; j < opt->dim; j++) {
            opt->population[curr_idx].position[j] += 
                fl * (opt->population[mother_idx].position[j] - opt->population[curr_idx].position[j]);
        }
    }
    enforce_bound_constraints(opt);
    free(sort_indices);
    free(mother_lib);
}

// Update individual and global best solutions
void update_best_solutions(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        double new_fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = new_fitness;
        if (new_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = new_fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
}

// Main Optimization Function
void CSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    initialize_population_cso(opt, objective_function);

    int *sort_indices = NULL;
    int *mate = NULL;
    int *mother_lib = NULL;
    int *mother_indices = NULL;

    for (int t = 0; t < opt->max_iter; t++) {
        // Update hierarchy every UPDATE_FREQ iterations or at start
        if (t % UPDATE_FREQ == 0 || t == 0) {
            free(sort_indices);
            free(mate);
            free(mother_lib);
            free(mother_indices);

            sort_indices = (int *)malloc(opt->population_size * sizeof(int));
            for (int i = 0; i < opt->population_size; i++) {
                sort_indices[i] = i;
            }
            for (int i = 0; i < opt->population_size - 1; i++) {
                for (int j = i + 1; j < opt->population_size; j++) {
                    if (opt->population[sort_indices[j]].fitness < opt->population[sort_indices[i]].fitness) {
                        int temp = sort_indices[i];
                        sort_indices[i] = sort_indices[j];
                        sort_indices[j] = temp;
                    }
                }
            }

            int hen_num = (int)(opt->population_size * HEN_RATIO);
            int mother_num = (int)(hen_num * MOTHER_RATIO);
            int rooster_num = (int)(opt->population_size * ROOSTER_RATIO);
            int chick_num = opt->population_size - rooster_num - hen_num;

            mate = (int *)malloc(hen_num * sizeof(int));
            randperm_f(rooster_num, hen_num, mate);

            mother_lib = (int *)malloc(mother_num * sizeof(int));
            randperm_f(hen_num, mother_num, mother_lib);
            for (int i = 0; i < mother_num; i++) {
                mother_lib[i] += rooster_num;
            }

            mother_indices = (int *)malloc(chick_num * sizeof(int));
            for (int i = 0; i < chick_num; i++) {
                mother_indices[i] = mother_lib[(int)(rand_double(0.0, 1.0) * mother_num)];
            }
        }

        update_roosters(opt);
        update_hens(opt);
        update_chicks(opt);
        update_best_solutions(opt, objective_function);

        // Log iteration (similar to Python print)
        printf("Iteration %d: Best Value = %f\n", t + 1, opt->best_solution.fitness);
    }

    free(sort_indices);
    free(mate);
    free(mother_lib);
    free(mother_indices);
}
