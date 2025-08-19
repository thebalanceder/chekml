#include "EA.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// === Utility Functions ===
static double random_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// === Memory Management ===

static void allocate_population(Solution *population, int pop_size, int dim) {
    for (int i = 0; i < pop_size; i++) {
        population[i].position = malloc(sizeof(double) * dim);
        if (population[i].position == NULL) {
            printf("ERROR: Memory allocation failed for population[%d].position\n", i);
            exit(1);
        }
        memset(population[i].position, 0, sizeof(double) * dim);
        population[i].fitness = DBL_MAX;
    }
}

static void free_population(Solution *population, int pop_size) {
    for (int i = 0; i < pop_size; i++) {
        if (population[i].position != NULL) {
            free(population[i].position);
            population[i].position = NULL;
        }
    }
}

// === Core EA Functions ===

void initialize_population_ea(Optimizer *opt, ObjectiveFunction objective_function) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int d = 0; d < opt->dim; d++) {
            double min_bound = opt->bounds[2 * d];
            double max_bound = opt->bounds[2 * d + 1];
            opt->population[i].position[d] = random_uniform(min_bound, max_bound);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

static int tournament_selection(Optimizer *opt, int *ranks, double *distances) {
    int idx1 = rand() % opt->population_size;
    int idx2 = rand() % opt->population_size;
    if (ranks[idx1] < ranks[idx2]) return idx1;
    if (ranks[idx1] > ranks[idx2]) return idx2;
    return (distances[idx1] > distances[idx2]) ? idx1 : idx2;
}

static void sbx_crossover(Optimizer *opt, int idx1, int idx2, Solution *child1, Solution *child2) {
    double *p1 = opt->population[idx1].position;
    double *p2 = opt->population[idx2].position;

    for (int i = 0; i < opt->dim; i++) {
        double u = random_uniform(0.0, 1.0);
        double beta = (u <= 0.5)
            ? pow(2 * u, 1.0 / (EA_MU + 1))
            : pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (EA_MU + 1));

        child1->position[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i]);
        child2->position[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i]);

        child1->position[i] = fmin(fmax(child1->position[i], opt->bounds[2 * i]), opt->bounds[2 * i + 1]);
        child2->position[i] = fmin(fmax(child2->position[i], opt->bounds[2 * i]), opt->bounds[2 * i + 1]);
    }
}

static void polynomial_mutation(Optimizer *opt, Solution *child) {
    for (int i = 0; i < opt->dim; i++) {
        double r = random_uniform(0.0, 1.0);
        double delta = (r < 0.5)
            ? pow(2 * r, 1.0 / (EA_MUM + 1)) - 1.0
            : 1.0 - pow(2 * (1.0 - r), 1.0 / (EA_MUM + 1));

        child->position[i] += delta;
        child->position[i] = fmin(fmax(child->position[i], opt->bounds[2 * i]), opt->bounds[2 * i + 1]);
    }
}

void evaluate_population_ea(Optimizer *opt, ObjectiveFunction objective_function) {
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

static void fast_non_dominated_sort(Optimizer *opt, int *fronts, int *ranks, double *distances) {
    for (int i = 0; i < opt->population_size; i++) {
        ranks[i] = 0;
        distances[i] = 0.0;
    }
}

static void update_best_solution(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * opt->dim);
        }
    }
}

void EA_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    opt->population = malloc(sizeof(Solution) * opt->population_size);
    if (opt->population == NULL) {
        printf("ERROR: Memory allocation failed for population\n");
        exit(1);
    }
    allocate_population(opt->population, opt->population_size, opt->dim);

    opt->best_solution.position = malloc(sizeof(double) * opt->dim);
    if (opt->best_solution.position == NULL) {
        printf("ERROR: Memory allocation failed for best_solution.position\n");
        exit(1);
    }
    opt->best_solution.fitness = DBL_MAX;

    initialize_population_ea(opt, objective_function);

    for (int iter = 0; iter < EA_GENERATIONS; iter++) {
        int *fronts = malloc(sizeof(int) * opt->population_size);
        int *ranks = malloc(sizeof(int) * opt->population_size);
        double *distances = malloc(sizeof(double) * opt->population_size);
        fast_non_dominated_sort(opt, fronts, ranks, distances);

        Solution *offspring = malloc(sizeof(Solution) * opt->population_size);
        allocate_population(offspring, opt->population_size, opt->dim);

        for (int i = 0; i < opt->population_size / 2; i++) {
            int p1 = tournament_selection(opt, ranks, distances);
            int p2 = tournament_selection(opt, ranks, distances);
            sbx_crossover(opt, p1, p2, &offspring[2 * i], &offspring[2 * i + 1]);
            polynomial_mutation(opt, &offspring[2 * i]);
            polynomial_mutation(opt, &offspring[2 * i + 1]);
        }

        for (int i = 0; i < opt->population_size; i++) {
            offspring[i].fitness = objective_function(offspring[i].position);
            memcpy(opt->population[i].position, offspring[i].position, sizeof(double) * opt->dim);
            opt->population[i].fitness = offspring[i].fitness;
        }

        update_best_solution(opt);
        printf("EA|%5d -----> %9.16f\n", iter + 1, opt->best_solution.fitness);

        free(fronts);
        free(ranks);
        free(distances);
        free_population(offspring, opt->population_size);
        free(offspring);
    }
}

void EA_free(Optimizer *opt) {
    if (opt->population != NULL) {
        free_population(opt->population, opt->population_size);
        free(opt->population);
        opt->population = NULL;
    }
    if (opt->best_solution.position != NULL) {
        free(opt->best_solution.position);
        opt->best_solution.position = NULL;
    }
}