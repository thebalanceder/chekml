#include "SPO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAX_ITER 100
#define PAINT_FACTOR 0.1
#define INF 1e30
#define MAX_DIM 128  // Adjust based on your maximum dimension size

// Fast RNG using Xoshiro or PCG (You could replace with your RNG of choice)
double rand_uniform(double min, double max) {
    // Simple linear congruential generator for fast random number generation
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// Bound with restrict for vectorization
static inline void bound(double* restrict position, const double* restrict LB, const double* restrict UB, int dim) {
    for (int i = 0; i < dim; i++) {
        if (position[i] > UB[i]) position[i] = UB[i];
        else if (position[i] < LB[i]) position[i] = LB[i];
    }
}

// Compare for qsort
int compare_fitness(const void* a, const void* b) {
    double f1 = ((Solution*)a)->fitness;
    double f2 = ((Solution*)b)->fitness;
    return (f1 > f2) - (f1 < f2);
}

void evaluate_population_spo(Optimizer* opt, ObjectiveFunction objective_function) {
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

void update_population_spo(Optimizer* opt, ObjectiveFunction objective_function) {
    int pop_size = opt->population_size;
    int dim = opt->dim;
    int N1 = pop_size / 3, N2 = pop_size / 3;
    int N3 = pop_size - N1 - N2;

    qsort(opt->population, pop_size, sizeof(Solution), compare_fitness);

    Solution* Group1 = opt->population;
    Solution* Group2 = &opt->population[N1];
    Solution* Group3 = &opt->population[N1 + N2];

    const double* restrict bounds = opt->bounds;
    double LB[dim], UB[dim];
    for (int d = 0; d < dim; d++) {
        LB[d] = bounds[2 * d];
        UB[d] = bounds[2 * d + 1];
    }

    // Reuse the buffer for temp positions
    static double temp_positions[4][MAX_DIM];  // Adjust MAX_DIM based on your dimension size

    #pragma omp parallel for
    for (int ind = 0; ind < pop_size; ind++) {
        double* current = opt->population[ind].position;

        // Complement Combination
        int id1 = rand() % N1, id2 = rand() % N3;
        for (int d = 0; d < dim; d++) {
            temp_positions[0][d] = current[d] + rand_uniform(0, 1) * 
                                   (Group1[id1].position[d] - Group3[id2].position[d]);
        }

        // Analog Combination (identity)
        memcpy(temp_positions[1], current, sizeof(double) * dim);

        // Triangle Combination
        id1 = rand() % N1; id2 = rand() % N2; int id3 = rand() % N3;
        for (int d = 0; d < dim; d++) {
            double avg = (Group1[id1].position[d] + Group2[id2].position[d] + Group3[id3].position[d]) / 3.0;
            temp_positions[2][d] = current[d] + rand_uniform(0, 1) * avg;
        }

        // Rectangle Combination
        id1 = rand() % N1; id2 = rand() % N2; id3 = rand() % N3; int id4 = rand() % pop_size;
        for (int d = 0; d < dim; d++) {
            double r1 = rand_uniform(0, 1);
            double r2 = rand_uniform(0, 1);
            double r3 = rand_uniform(0, 1);
            double r4 = rand_uniform(0, 1);
            temp_positions[3][d] = current[d] +
                r1 * Group1[id1].position[d] +
                r2 * Group2[id2].position[d] +
                r3 * Group3[id3].position[d] +
                r4 * opt->population[id4].position[d];
        }

        for (int i = 0; i < 4; i++) {
            bound(temp_positions[i], LB, UB, dim);
            double fitness = objective_function(temp_positions[i]);
            if (fitness < opt->population[ind].fitness) {
                memcpy(current, temp_positions[i], sizeof(double) * dim);
                opt->population[ind].fitness = fitness;
            }
        }
    }
}

void SPO_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Initialize population with random positions
    for (int i = 0; i < pop_size; i++) {
        opt->population[i].position = (double*)aligned_alloc(32, sizeof(double) * dim);
        for (int d = 0; d < dim; d++) {
            opt->population[i].position[d] = rand_uniform(opt->bounds[2 * d], opt->bounds[2 * d + 1]);
        }
        opt->population[i].fitness = INF;
    }

    opt->best_solution.fitness = INF;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        evaluate_population_spo(opt, objective_function);
        update_population_spo(opt, objective_function);

        // Optional: you can add an enforce_bounds function here if you have any additional constraints

        for (int i = 0; i < pop_size; i++) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * dim);
                opt->best_solution.fitness = opt->population[i].fitness;
            }
        }

        printf("Iteration %3d: Best Fitness = %.10f\n", iter, opt->best_solution.fitness);
    }
}
