#include "CMOA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Helper function to find the index of the closest solution to solution i
int find_closest_solution(Optimizer *opt, int i) {
    double min_distance = INFINITY;
    int closest_idx = i;
    for (int k = 0; k < opt->population_size; k++) {
        if (k == i) continue;
        double distance = 0.0;
        for (int j = 0; j < opt->dim; j++) {
            double diff = opt->population[k].position[j] - opt->population[i].position[j];
            distance += fabs(diff);
        }
        if (distance < min_distance) {
            min_distance = distance;
            closest_idx = k;
        }
    }
    return closest_idx;
}

// Genetic Recombination Phase
void genetic_recombination(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        if (rand_double(0.0, 1.0) < CMOA_CROSSOVER_RATE) {
            int closest_idx = find_closest_solution(opt, i);
            for (int j = 0; j < opt->dim; j++) {
                double diff = opt->population[closest_idx].position[j] - opt->population[i].position[j];
                opt->population[i].position[j] += rand_double(0.0, 1.0) * diff;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Cross-Activation Phase
void cross_activation(Optimizer *opt) {
    if (opt->best_solution.fitness != INFINITY) {  // Ensure best_solution is initialized
        for (int i = 0; i < opt->population_size; i++) {
            for (int j = 0; j < opt->dim; j++) {
                double diff = opt->best_solution.position[j] - opt->population[i].position[j];
                opt->population[i].position[j] += rand_double(0.0, 1.0) * diff;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Incremental Reactivation Phase
void incremental_reactivation(Optimizer *opt, int t) {
    double T = (double)opt->max_iter;
    double evolutionary_operator = cos(M_PI * ((double)t / T));
    if (opt->best_solution.fitness != INFINITY) {  // Ensure best_solution is initialized
        for (int i = 0; i < opt->population_size; i++) {
            for (int j = 0; j < opt->dim; j++) {
                double diff = opt->best_solution.position[j] - opt->population[i].position[j];
                opt->population[i].position[j] += evolutionary_operator * rand_double(0.0, 1.0) * diff;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Non-Genetic Mutation Phase
void non_genetic_mutation(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        if (rand_double(0.0, 1.0) < CMOA_MUTATION_RATE) {
            for (int j = 0; j < opt->dim; j++) {
                double range = opt->bounds[2 * j + 1] - opt->bounds[2 * j];
                opt->population[i].position[j] += rand_double(-CMOA_MUTATION_SCALE, CMOA_MUTATION_SCALE) * range;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Genotypic Mixing Phase
void genotypic_mixing(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        // Randomly select two distinct indices
        int idx1 = rand() % opt->population_size;
        int idx2;
        do {
            idx2 = rand() % opt->population_size;
        } while (idx2 == idx1);
        for (int j = 0; j < opt->dim; j++) {
            double diff = opt->population[idx1].position[j] - opt->population[idx2].position[j];
            opt->population[i].position[j] += rand_double(0.0, 1.0) * diff;
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void CMOA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize random seed
    srand((unsigned int)time(NULL));

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

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        genetic_recombination(opt);
        cross_activation(opt);
        incremental_reactivation(opt, iter);
        non_genetic_mutation(opt);
        genotypic_mixing(opt);

        // Evaluate population and update best solution
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
        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
