/* BSA.c - Implementation of Backtracking Search Algorithm */
#include "BSA.h"
#include <time.h>

static double get_scale_factor() {
    return 3.0 * rand_double(-1.0, 1.0);  // Brownian-walk based factor
}

static void generate_population(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
    }
}

static void boundary_control(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double r = rand_double(0.0, 1.0);
            double k = rand_double(0.0, 1.0);
            if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                opt->population[i].position[j] = (r < k) ? opt->bounds[2 * j] : rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
            } else if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                opt->population[i].position[j] = (r < k) ? opt->bounds[2 * j + 1] : rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
            }
        }
    }
}

void BSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    Solution *historical_pop = malloc(sizeof(Solution) * opt->population_size);
    for (int i = 0; i < opt->population_size; i++) {
        historical_pop[i].position = malloc(sizeof(double) * opt->dim);
        historical_pop[i].fitness = INFINITY;
    }

    generate_population(opt);
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
    generate_population(opt); // For historical_pop
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            historical_pop[i].position[j] = opt->population[i].position[j];
        }
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        if (rand_double(0.0, 1.0) < rand_double(0.0, 1.0)) {
            for (int i = 0; i < opt->population_size; i++) {
                for (int j = 0; j < opt->dim; j++) {
                    historical_pop[i].position[j] = opt->population[i].position[j];
                }
            }
        }

        // Shuffle historical_pop
        for (int i = opt->population_size - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            Solution temp = historical_pop[i];
            historical_pop[i] = historical_pop[j];
            historical_pop[j] = temp;
        }

        double F = get_scale_factor();
        int **map = malloc(sizeof(int*) * opt->population_size);
        for (int i = 0; i < opt->population_size; i++) {
            map[i] = calloc(opt->dim, sizeof(int));
            if (rand_double(0.0, 1.0) < rand_double(0.0, 1.0)) {
                int count = (int)ceil(BSA_DIM_RATE * rand_double(0.0, 1.0) * opt->dim);
                for (int c = 0; c < count; c++) {
                    map[i][rand() % opt->dim] = 1;
                }
            } else {
                map[i][rand() % opt->dim] = 1;
            }
        }

        for (int i = 0; i < opt->population_size; i++) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] += map[i][j] * F * (historical_pop[i].position[j] - opt->population[i].position[j]);
            }
        }

        boundary_control(opt);

        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            if (new_fitness < opt->population[i].fitness) {
                opt->population[i].fitness = new_fitness;
                if (new_fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = new_fitness;
                    for (int j = 0; j < opt->dim; j++) {
                        opt->best_solution.position[j] = opt->population[i].position[j];
                    }
                }
            }
        }

        printf("BSA|%5d -----> %9.16f\n", iter + 1, opt->best_solution.fitness);

        for (int i = 0; i < opt->population_size; i++) free(map[i]);
        free(map);
        enforce_bound_constraints(opt);
    }

    for (int i = 0; i < opt->population_size; i++) free(historical_pop[i].position);
    free(historical_pop);
}