#include "CSO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <float.h>

// Structure to hold pre-allocated buffers
typedef struct {
    int *sort_indices;
    int *mate;
    int *mother_lib;
    int *temp_perm;
} CSO_Buffers;

// Global variable to pass Optimizer to qsort
static Optimizer *g_opt = NULL;

// Comparison function for qsort
static int compare_fitness(const void *a, const void *b) {
    int idx_a = *(int *)a;
    int idx_b = *(int *)b;
    return (g_opt->population[idx_a].fitness > g_opt->population[idx_b].fitness) - 
           (g_opt->population[idx_a].fitness < g_opt->population[idx_b].fitness);
}

// Fast random double generator
static double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Random integer excluding tabu value
static int randi_tabu(int min_val, int max_val, int tabu) {
    int temp;
    do {
        temp = min_val + (int)((max_val - min_val + 1) * rand_double(0.0, 1.0));
    } while (temp == tabu);
    return temp;
}

// Optimized permutation generator
static void randperm_f(int range_size, int dim, int *result, int *temp) {
    for (int i = 0; i < range_size; i++) {
        temp[i] = i + 1;
    }
    for (int i = range_size - 1; i > 0; i--) {
        int j = (int)(rand_double(0.0, 1.0) * (i + 1));
        int swap = temp[i];
        temp[i] = temp[j];
        temp[j] = swap;
    }
    for (int i = 0; i < dim; i++) {
        result[i] = (i < range_size) ? temp[i] : (1 + (int)(rand_double(0.0, 1.0) * range_size));
    }
}

// Initialize population
void initialize_population_cso(Optimizer *opt, double (*objective_function)(double *)) {
    // Validate bounds
    for (int j = 0; j < opt->dim; j++) {
        if (opt->bounds[2*j] > opt->bounds[2*j+1]) {
            fprintf(stderr, "Error: Invalid bounds at dimension %d: [%f, %f]\n",
                    j, opt->bounds[2*j], opt->bounds[2*j+1]);
            exit(1);
        }
    }

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (isnan(opt->population[i].fitness) || isinf(opt->population[i].fitness)) {
            fprintf(stderr, "Warning: Invalid initial fitness at population[%d]: %f, position=[", i, opt->population[i].fitness);
            for (int j = 0; j < opt->dim; j++) {
                fprintf(stderr, "%f ", opt->population[i].position[j]);
            }
            fprintf(stderr, "]\n");
            opt->population[i].fitness = DBL_MAX;
        }
    }
    enforce_bound_constraints(opt);

    int best_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[best_idx].fitness) {
            best_idx = i;
        }
    }
    opt->best_solution.fitness = opt->population[best_idx].fitness;
    memcpy(opt->best_solution.position, opt->population[best_idx].position, opt->dim * sizeof(double));
}

// Update roosters
void update_roosters(Optimizer *opt, int *sort_indices, int rooster_num) {
    for (int i = 0; i < rooster_num; i++) {
        int curr_idx = sort_indices[i];
        int another_rooster = randi_tabu(1, rooster_num, i + 1);
        int another_idx = sort_indices[another_rooster - 1];

        double sigma = 1.0;
        double curr_fitness = opt->population[curr_idx].fitness;
        double other_fitness = opt->population[another_idx].fitness;
        if (curr_fitness > other_fitness && isfinite(curr_fitness) && isfinite(other_fitness)) {
            double diff = other_fitness - curr_fitness;
            double denom = fabs(curr_fitness) + 2.2e-16;
            if (fabs(diff / denom) < 100.0) {
                sigma = exp(diff / denom);
            }
        }

        for (int j = 0; j < opt->dim; j++) {
            double factor = 1.0 + sigma * rand_double(-3.0, 3.0);
            opt->population[curr_idx].position[j] *= factor;
            if (isnan(opt->population[curr_idx].position[j]) || isinf(opt->population[curr_idx].position[j])) {
                fprintf(stderr, "Warning: Invalid position at rooster[%d][%d]: %f\n", curr_idx, j, opt->population[curr_idx].position[j]);
                opt->population[curr_idx].position[j] = opt->bounds[2 * j];
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Update hens
void update_hens(Optimizer *opt, int *sort_indices, int rooster_num, int hen_num, int *mate) {
    for (int i = rooster_num; i < rooster_num + hen_num; i++) {
        int curr_idx = sort_indices[i];
        int mate_idx = sort_indices[mate[i - rooster_num] - 1];
        int other = randi_tabu(1, i, mate[i - rooster_num]);
        int other_idx = sort_indices[other - 1];

        double curr_fitness = opt->population[curr_idx].fitness;
        double mate_fitness = opt->population[mate_idx].fitness;
        double other_fitness = opt->population[other_idx].fitness;

        double c1 = 1.0, c2 = 1.0;
        if (isfinite(curr_fitness) && isfinite(mate_fitness) && isfinite(other_fitness)) {
            double diff1 = curr_fitness - mate_fitness;
            double denom1 = fabs(curr_fitness) + 2.2e-16;
            if (fabs(diff1 / denom1) < 100.0) {
                c1 = exp(diff1 / denom1);
            }
            double diff2 = other_fitness - curr_fitness;
            if (fabs(diff2) < 100.0) {
                c2 = exp(diff2);
            }
        }

        for (int j = 0; j < opt->dim; j++) {
            double delta1 = c1 * rand_double(0.0, 1.0) * (opt->population[mate_idx].position[j] - opt->population[curr_idx].position[j]);
            double delta2 = c2 * rand_double(0.0, 1.0) * (opt->population[other_idx].position[j] - opt->population[curr_idx].position[j]);
            opt->population[curr_idx].position[j] += delta1 + delta2;
            if (isnan(opt->population[curr_idx].position[j]) || isinf(opt->population[curr_idx].position[j])) {
                fprintf(stderr, "Warning: Invalid position at hen[%d][%d]: %f\n", curr_idx, j, opt->population[curr_idx].position[j]);
                opt->population[curr_idx].position[j] = opt->bounds[2 * j];
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Update chicks
void update_chicks(Optimizer *opt, int *sort_indices, int rooster_num, int hen_num, int mother_num, int *mother_lib) {
    int chick_num = opt->population_size - rooster_num - hen_num;
    for (int i = rooster_num + hen_num; i < opt->population_size; i++) {
        int curr_idx = sort_indices[i];
        int mother_idx = sort_indices[mother_lib[(int)(rand_double(0.0, 1.0) * mother_num)] - 1];
        double fl = 0.5 + 0.4 * rand_double(0.0, 1.0);

        for (int j = 0; j < opt->dim; j++) {
            double delta = fl * (opt->population[mother_idx].position[j] - opt->population[curr_idx].position[j]);
            opt->population[curr_idx].position[j] += delta;
            if (isnan(opt->population[curr_idx].position[j]) || isinf(opt->population[curr_idx].position[j])) {
                fprintf(stderr, "Warning: Invalid position at chick[%d][%d]: %f\n", curr_idx, j, opt->population[curr_idx].position[j]);
                opt->population[curr_idx].position[j] = opt->bounds[2 * j];
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Update best solutions
void update_best_solutions(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        double new_fitness = objective_function(opt->population[i].position);
        if (isnan(new_fitness) || isinf(new_fitness)) {
            fprintf(stderr, "Warning: Invalid fitness at population[%d]: %f, position=[", i, new_fitness);
            for (int j = 0; j < opt->dim; j++) {
                fprintf(stderr, "%f ", opt->population[i].position[j]);
            }
            fprintf(stderr, "]\n");
            new_fitness = DBL_MAX;
        }
        opt->population[i].fitness = new_fitness;
        if (new_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = new_fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
        }
    }
}

// Main Optimization Function
void CSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand((unsigned int)time(NULL));

    // Pre-compute constants
    int rooster_num = (int)(opt->population_size * ROOSTER_RATIO);
    int hen_num = (int)(opt->population_size * HEN_RATIO);
    int mother_num = (int)(hen_num * MOTHER_RATIO);
    int chick_num = opt->population_size - rooster_num - hen_num;

    // Allocate buffers
    CSO_Buffers buffers = {
        .sort_indices = (int *)malloc(opt->population_size * sizeof(int)),
        .mate = (int *)malloc(hen_num * sizeof(int)),
        .mother_lib = (int *)malloc(mother_num * sizeof(int)),
        .temp_perm = (int *)malloc(opt->population_size * sizeof(int))
    };
    if (!buffers.sort_indices || !buffers.mate || !buffers.mother_lib || !buffers.temp_perm) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }

    initialize_population_cso(opt, objective_function);

    for (int t = 0; t < opt->max_iter; t++) {
        if (t % UPDATE_FREQ == 0 || t == 0) {
            for (int i = 0; i < opt->population_size; i++) {
                buffers.sort_indices[i] = i;
            }
            g_opt = opt;
            qsort(buffers.sort_indices, opt->population_size, sizeof(int), compare_fitness);
            g_opt = NULL;

            randperm_f(rooster_num, hen_num, buffers.mate, buffers.temp_perm);
            randperm_f(hen_num, mother_num, buffers.mother_lib, buffers.temp_perm);
            for (int i = 0; i < mother_num; i++) {
                buffers.mother_lib[i] += rooster_num;
            }
        }

        update_roosters(opt, buffers.sort_indices, rooster_num);
        update_hens(opt, buffers.sort_indices, rooster_num, hen_num, buffers.mate);
        update_chicks(opt, buffers.sort_indices, rooster_num, hen_num, mother_num, buffers.mother_lib);
        update_best_solutions(opt, objective_function);

        if (isnan(opt->best_solution.fitness) || isinf(opt->best_solution.fitness)) {
            fprintf(stderr, "Warning: Best solution invalid at iteration %d: fitness=%f\n", t + 1, opt->best_solution.fitness);
        }
        printf("Iteration %d: Best Value = %f\n", t + 1, opt->best_solution.fitness);
    }

    free(buffers.sort_indices);
    free(buffers.mate);
    free(buffers.mother_lib);
    free(buffers.temp_perm);
}
