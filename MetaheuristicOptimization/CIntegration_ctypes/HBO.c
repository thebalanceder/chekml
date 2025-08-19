#include "HBO.h"
#include "generaloptimizer.h"
#include <float.h>
#include <time.h>

static double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// ⚙️ Initialization of population with random positions
static void initialize_population(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = rand_double(lb, ub);
        }
        opt->population[i].fitness = INFINITY;
    }
}

// 📌 Enforce bounds on each individual
static void enforce_bounds_individual(double *position, int dim, const double *bounds) {
    for (int j = 0; j < dim; j++) {
        if (position[j] < bounds[2 * j]) position[j] = bounds[2 * j];
        else if (position[j] > bounds[2 * j + 1]) position[j] = bounds[2 * j + 1];
    }
}

// 🧠 Main Optimization Logic
void HBO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand((unsigned int)time(NULL));

    initialize_population(opt);

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
        for (int i = 0; i < opt->population_size; i++) {
            double *individual = opt->population[i].position;

            int idx[3];
            do {
                idx[0] = rand() % opt->population_size;
                idx[1] = rand() % opt->population_size;
                idx[2] = rand() % opt->population_size;
            } while (idx[0] == idx[1] || idx[1] == idx[2] || idx[0] == idx[2]);

            double *a = opt->population[idx[0]].position;
            double *b = opt->population[idx[1]].position;
            double *c = opt->population[idx[2]].position;

            double r1 = rand_double(0.0, 1.0);
            double r2 = rand_double(0.0, 1.0);

            double heap_candidate[opt->dim];
            for (int j = 0; j < opt->dim; j++) {
                heap_candidate[j] = a[j] + r1 * (b[j] - c[j]);
            }

            // Optional dimension mixing (heap-like)
            int col_idx = rand() % opt->dim;
            double trial[opt->dim];
            for (int j = 0; j < opt->dim; j++) {
                trial[j] = individual[j];
            }
            trial[col_idx] = heap_candidate[col_idx];

            enforce_bounds_individual(trial, opt->dim, opt->bounds);

            double trial_fitness = objective_function(trial);
            if (trial_fitness < opt->population[i].fitness) {
                opt->population[i].fitness = trial_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    individual[j] = trial[j];
                }

                if (trial_fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = trial_fitness;
                    for (int j = 0; j < opt->dim; j++) {
                        opt->best_solution.position[j] = trial[j];
                    }
                }
            }
        }
    }
}

