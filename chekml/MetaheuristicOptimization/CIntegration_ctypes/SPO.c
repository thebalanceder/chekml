#include "SPO.h"
#include "generaloptimizer.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Constants
#define MAX_ITER 100
#define PAINT_FACTOR 0.1
#define INF 1e30

// Random double in range [min, max]
double rand_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// Bound a position within [LB, UB]
void bound(double* position, double* LB, double* UB, int dim) {
    for (int i = 0; i < dim; i++) {
        if (position[i] > UB[i]) position[i] = UB[i];
        if (position[i] < LB[i]) position[i] = LB[i];
    }
}

// Evaluate the entire population
void evaluate_population_spo(Optimizer* opt, ObjectiveFunction objective_function) {
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// SPO movement operator
void update_population_spo(Optimizer* opt, ObjectiveFunction objective_function) {
    int N1stColors = opt->population_size / 3;
    int N2ndColors = opt->population_size / 3;
    int N3rdColors = opt->population_size - N1stColors - N2ndColors;

    // Sort population by fitness (ascending)
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = i + 1; j < opt->population_size; j++) {
            if (opt->population[i].fitness > opt->population[j].fitness) {
                Solution temp = opt->population[i];
                opt->population[i] = opt->population[j];
                opt->population[j] = temp;
            }
        }
    }

    Solution* Group1st = &opt->population[0];
    Solution* Group2nd = &opt->population[N1stColors];
    Solution* Group3rd = &opt->population[N1stColors + N2ndColors];

    double* LB = (double*)malloc(sizeof(double) * opt->dim);
    double* UB = (double*)malloc(sizeof(double) * opt->dim);
    for (int d = 0; d < opt->dim; d++) {
        LB[d] = opt->bounds[2 * d];
        UB[d] = opt->bounds[2 * d + 1];
    }

    for (int ind = 0; ind < opt->population_size; ind++) {
        Solution newColors[4];
        for (int i = 0; i < 4; i++) {
            newColors[i].position = (double*)malloc(sizeof(double) * opt->dim);
        }

        // --- Complement Combination ---
        int Id1 = rand() % N1stColors;
        int Id2 = rand() % N3rdColors;
        for (int d = 0; d < opt->dim; d++) {
            newColors[0].position[d] = opt->population[ind].position[d] +
                rand_uniform(0, 1) * (Group1st[Id1].position[d] - Group3rd[Id2].position[d]);
        }

        // --- Analog Combination (no-op for now) ---
        for (int d = 0; d < opt->dim; d++) {
            newColors[1].position[d] = opt->population[ind].position[d];
        }

        // --- Triangle Combination ---
        Id1 = rand() % N1stColors;
        Id2 = rand() % N2ndColors;
        int Id3 = rand() % N3rdColors;
        for (int d = 0; d < opt->dim; d++) {
            newColors[2].position[d] = opt->population[ind].position[d] +
                rand_uniform(0, 1) * ((Group1st[Id1].position[d] +
                                       Group2nd[Id2].position[d] +
                                       Group3rd[Id3].position[d]) / 3.0);
        }

        // --- Rectangle Combination ---
        Id1 = rand() % N1stColors;
        Id2 = rand() % N2ndColors;
        Id3 = rand() % N3rdColors;
        int Id4 = rand() % opt->population_size;
        for (int d = 0; d < opt->dim; d++) {
            newColors[3].position[d] = opt->population[ind].position[d] +
                rand_uniform(0, 1) * Group1st[Id1].position[d] +
                rand_uniform(0, 1) * Group2nd[Id2].position[d] +
                rand_uniform(0, 1) * Group3rd[Id3].position[d] +
                rand_uniform(0, 1) * opt->population[Id4].position[d];
        }

        // --- Evaluate new candidates and update if better ---
        for (int i = 0; i < 4; i++) {
            bound(newColors[i].position, LB, UB, opt->dim);
            newColors[i].fitness = objective_function(newColors[i].position);
            if (newColors[i].fitness < opt->population[ind].fitness) {
                memcpy(opt->population[ind].position, newColors[i].position, sizeof(double) * opt->dim);
                opt->population[ind].fitness = newColors[i].fitness;
            }
            free(newColors[i].position);
        }
    }

    free(LB);
    free(UB);
}

// SPO Main Optimization Routine
void SPO_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    // Initialize population
	for (int i = 0; i < opt->population_size; i++) {
		opt->population[i].position = (double*)malloc(sizeof(double) * opt->dim);
		for (int d = 0; d < opt->dim; d++) {
			double min_bound = opt->bounds[2 * d];
			double max_bound = opt->bounds[2 * d + 1];
			opt->population[i].position[d] = rand_uniform(min_bound, max_bound);
		}
		opt->population[i].fitness = INF;
	}

    // Initialize best solution
    opt->best_solution.fitness = INF;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        evaluate_population_spo(opt, objective_function);
        update_population_spo(opt, objective_function);
        enforce_bound_constraints(opt);

        // Update best_solution
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * opt->dim);
                opt->best_solution.fitness = opt->population[i].fitness;
            }
        }

        printf("Iteration %d: Best Fitness = %f\n", iter, opt->best_solution.fitness);
    }
}
