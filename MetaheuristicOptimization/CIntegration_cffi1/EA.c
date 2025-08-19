#include "EA.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <immintrin.h>  // SIMD

// === Fast Random Generator ===
static inline double fast_random_uniform() {
    return (double)rand() / (double)RAND_MAX;
}

static inline double random_uniform(double min, double max) {
    return min + fast_random_uniform() * (max - min);
}

// === Memory Management ===
static inline void allocate_population(Solution *population, int pop_size, int dim) {
    for (int i = 0; i < pop_size; ++i) {
        if (posix_memalign((void**)&population[i].position, 64, dim * sizeof(double)) != 0) {
            fprintf(stderr, "ERROR: Aligned allocation failed for population[%d].position\n", i);
            exit(EXIT_FAILURE);
        }
        memset(population[i].position, 0, dim * sizeof(double));
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
        double *pos = opt->population[i].position;
        for (int d = 0; d < opt->dim; ++d) {
            pos[d] = random_uniform(opt->bounds[2 * d], opt->bounds[2 * d + 1]);
        }
        opt->population[i].fitness = objective_function(pos);
    }
}

static inline int tournament_selection(const Optimizer *opt, const int *ranks, const double *distances) {
    int a = rand() % opt->population_size;
    int b = rand() % opt->population_size;
    if (ranks[a] < ranks[b]) return a;
    if (ranks[a] > ranks[b]) return b;
    return distances[a] > distances[b] ? a : b;
}

static inline void sbx_crossover(const Optimizer *opt, int p1, int p2, Solution *c1, Solution *c2) {
    const double *x1 = opt->population[p1].position;
    const double *x2 = opt->population[p2].position;
    double *pos1 = c1->position;
    double *pos2 = c2->position;

    for (int i = 0; i < opt->dim; ++i) {
        double u = fast_random_uniform();
        double beta = (u <= 0.5)
            ? pow(2.0 * u, 1.0 / (EA_MU + 1))
            : pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (EA_MU + 1));

        double min_b = opt->bounds[2 * i];
        double max_b = opt->bounds[2 * i + 1];

        double v1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i]);
        double v2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i]);

        pos1[i] = fmin(fmax(v1, min_b), max_b);
        pos2[i] = fmin(fmax(v2, min_b), max_b);
    }
}

static inline void polynomial_mutation(const Optimizer *opt, Solution *sol) {
    double *pos = sol->position;
    for (int i = 0; i < opt->dim; ++i) {
        double r = fast_random_uniform();
        double delta = (r < 0.5)
            ? pow(2.0 * r, 1.0 / (EA_MUM + 1)) - 1.0
            : 1.0 - pow(2.0 * (1.0 - r), 1.0 / (EA_MUM + 1));

        pos[i] += delta;
        pos[i] = fmin(fmax(pos[i], opt->bounds[2 * i]), opt->bounds[2 * i + 1]);
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
    (void)fronts;
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
    opt->population = (Solution *)aligned_alloc(64, sizeof(Solution) * opt->population_size);
    if (!opt->population) {
        fprintf(stderr, "ERROR: Allocation failed for population\n");
        exit(EXIT_FAILURE);
    }
    allocate_population(opt->population, opt->population_size, opt->dim);

    if (posix_memalign((void**)&opt->best_solution.position, 64, sizeof(double) * opt->dim) != 0) {
        fprintf(stderr, "ERROR: Allocation failed for best_solution.position\n");
        exit(EXIT_FAILURE);
    }
    opt->best_solution.fitness = DBL_MAX;

    initialize_population_ea(opt, objective_function);

    int *fronts = (int *)malloc(sizeof(int) * opt->population_size);
    int *ranks = (int *)malloc(sizeof(int) * opt->population_size);
    double *distances = (double *)malloc(sizeof(double) * opt->population_size);
    Solution *offspring = (Solution *)aligned_alloc(64, sizeof(Solution) * opt->population_size);
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
