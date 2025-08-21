#include "EA.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// === Utility Functions ===
static inline double random_uniform(double min, double max) {
    return min + ((double)rand() / (double)RAND_MAX) * (max - min);
}

// === Memory Management ===
static inline void allocate_population(Solution *population, int pop_size, int dim) {
    for (int i = 0; i < pop_size; ++i) {
        population[i].position = (double *)calloc(dim, sizeof(double));
        if (!population[i].position) {
            fprintf(stderr, "ERROR: Allocation failed for population[%d].position\n", i);
            exit(EXIT_FAILURE);
        }
        population[i].fitness = DBL_MAX;
    }
}

static inline void free_population(Solution *population, int pop_size) {
    for (int i = 0; i < pop_size; ++i) {
        free(population[i].position);
        population[i].position = NULL;
    }
}

// === Core EA Functions ===
void initialize_population_ea(Optimizer *opt, ObjectiveFunction objective_function) {
    for (int i = 0; i < opt->population_size; ++i) {
        for (int d = 0; d < opt->dim; ++d) {
            double min = opt->bounds[2 * d];
            double max = opt->bounds[2 * d + 1];
            opt->population[i].position[d] = random_uniform(min, max);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

static inline int tournament_selection(const Optimizer *opt, const int *ranks, const double *distances) {
    int a = rand() % opt->population_size;
    int b = rand() % opt->population_size;
    if (ranks[a] < ranks[b]) return a;
    if (ranks[a] > ranks[b]) return b;
    return (distances[a] > distances[b]) ? a : b;
}

static inline void sbx_crossover(const Optimizer *opt, int p1, int p2, Solution *c1, Solution *c2) {
    const double *x1 = opt->population[p1].position;
    const double *x2 = opt->population[p2].position;
    for (int i = 0; i < opt->dim; ++i) {
        double u = random_uniform(0.0, 1.0);
        double beta = (u <= 0.5)
            ? pow(2.0 * u, 1.0 / (EA_MU + 1))
            : pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (EA_MU + 1));

        double min_bound = opt->bounds[2 * i];
        double max_bound = opt->bounds[2 * i + 1];

        c1->position[i] = fmin(fmax(0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i]), min_bound), max_bound);
        c2->position[i] = fmin(fmax(0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i]), min_bound), max_bound);
    }
}

static inline void polynomial_mutation(const Optimizer *opt, Solution *sol) {
    for (int i = 0; i < opt->dim; ++i) {
        double r = random_uniform(0.0, 1.0);
        double delta = (r < 0.5)
            ? pow(2.0 * r, 1.0 / (EA_MUM + 1)) - 1.0
            : 1.0 - pow(2.0 * (1.0 - r), 1.0 / (EA_MUM + 1));

        sol->position[i] += delta;
        sol->position[i] = fmin(fmax(sol->position[i], opt->bounds[2 * i]), opt->bounds[2 * i + 1]);
    }
}

void evaluate_population_ea(Optimizer *opt, ObjectiveFunction objective_function) {
    for (int i = 0; i < opt->population_size; ++i) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

static inline void fast_non_dominated_sort(const Optimizer *opt, int *fronts, int *ranks, double *distances) {
    memset(ranks, 0, sizeof(int) * opt->population_size);
    memset(distances, 0, sizeof(double) * opt->population_size);
    (void)fronts; // Suppress unused parameter warning
}

static inline void update_best_solution(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; ++i) {
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * opt->dim);
        }
    }
}

void EA_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    opt->population = (Solution *)malloc(sizeof(Solution) * opt->population_size);
    if (!opt->population) {
        fprintf(stderr, "ERROR: Allocation failed for population\n");
        exit(EXIT_FAILURE);
    }
    allocate_population(opt->population, opt->population_size, opt->dim);

    opt->best_solution.position = (double *)malloc(sizeof(double) * opt->dim);
    if (!opt->best_solution.position) {
        fprintf(stderr, "ERROR: Allocation failed for best_solution.position\n");
        exit(EXIT_FAILURE);
    }
    opt->best_solution.fitness = DBL_MAX;

    initialize_population_ea(opt, objective_function);

    int *fronts = (int *)malloc(sizeof(int) * opt->population_size);
    int *ranks = (int *)malloc(sizeof(int) * opt->population_size);
    double *distances = (double *)malloc(sizeof(double) * opt->population_size);
    Solution *offspring = (Solution *)malloc(sizeof(Solution) * opt->population_size);
    allocate_population(offspring, opt->population_size, opt->dim);

    for (int iter = 0; iter < EA_GENERATIONS; ++iter) {
        fast_non_dominated_sort(opt, fronts, ranks, distances);

        for (int i = 0; i < opt->population_size / 2; ++i) {
            int p1 = tournament_selection(opt, ranks, distances);
            int p2 = tournament_selection(opt, ranks, distances);

            sbx_crossover(opt, p1, p2, &offspring[2 * i], &offspring[2 * i + 1]);
            polynomial_mutation(opt, &offspring[2 * i]);
            polynomial_mutation(opt, &offspring[2 * i + 1]);
        }

        for (int i = 0; i < opt->population_size; ++i) {
            offspring[i].fitness = objective_function(offspring[i].position);
            memcpy(opt->population[i].position, offspring[i].position, sizeof(double) * opt->dim);
            opt->population[i].fitness = offspring[i].fitness;
        }

        update_best_solution(opt);
        printf("EA|%5d -----> %9.16f\n", iter + 1, opt->best_solution.fitness);
    }

    free(fronts);
    free(ranks);
    free(distances);
    free_population(offspring, opt->population_size);
    free(offspring);
}

void EA_free(Optimizer *opt) {
    if (opt->population) {
        free_population(opt->population, opt->population_size);
        free(opt->population);
        opt->population = NULL;
    }
    if (opt->best_solution.position) {
        free(opt->best_solution.position);
        opt->best_solution.position = NULL;
    }
}
