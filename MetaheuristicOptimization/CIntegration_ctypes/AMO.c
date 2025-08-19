#include "AMO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Function to generate a normally distributed random number (Box-Muller transform)
double normal_rand(double mean, double stddev) {
    static int has_spare = 0;
    static double spare;
    
    if (has_spare) {
        has_spare = 0;
        return mean + stddev * spare;
    }
    
    has_spare = 1;
    double u, v, s;
    do {
        u = rand_double(-1.0, 1.0);
        v = rand_double(-1.0, 1.0);
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + stddev * u * s;
}

// Initialize Population
void initialize_population_amo(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Neighborhood Learning Phase
void neighborhood_learning_phase(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double FF = normal_rand(0.0, 1.0);
        int lseq[NEIGHBORHOOD_SIZE_AMO];
        
        // Define neighborhood based on index
        if (i == 0) {
            lseq[0] = opt->population_size - 2;
            lseq[1] = opt->population_size - 1;
            lseq[2] = i;
            lseq[3] = i + 1;
            lseq[4] = i + 2;
        } else if (i == 1) {
            lseq[0] = opt->population_size - 1;
            lseq[1] = i - 1;
            lseq[2] = i;
            lseq[3] = i + 1;
            lseq[4] = i + 2;
        } else if (i == opt->population_size - 2) {
            lseq[0] = i - 2;
            lseq[1] = i - 1;
            lseq[2] = i;
            lseq[3] = opt->population_size - 1;
            lseq[4] = 0;
        } else if (i == opt->population_size - 1) {
            lseq[0] = i - 2;
            lseq[1] = i - 1;
            lseq[2] = i;
            lseq[3] = 0;
            lseq[4] = 1;
        } else {
            lseq[0] = i - 2;
            lseq[1] = i - 1;
            lseq[2] = i;
            lseq[3] = i + 1;
            lseq[4] = i + 2;
        }
        
        // Random permutation of neighborhood indices
        for (int j = 0; j < NEIGHBORHOOD_SIZE_AMO; j++) {
            int temp = lseq[j];
            int idx = rand() % NEIGHBORHOOD_SIZE_AMO;
            lseq[j] = lseq[idx];
            lseq[idx] = temp;
        }
        
        // Select second element as exemplar
        int exemplar_idx = lseq[1];
        
        // Update position
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] += FF * (opt->population[exemplar_idx].position[j] - opt->population[i].position[j]);
        }
    }
    enforce_bound_constraints(opt);
}

// Global Migration Phase
void global_migration_phase(Optimizer *opt) {
    // Compute fitness-based probabilities
    double* probabilities = (double*)malloc(opt->population_size * sizeof(double));
    int* sort_indices = (int*)malloc(opt->population_size * sizeof(int));
    
    // Initialize sort_indices
    for (int i = 0; i < opt->population_size; i++) {
        sort_indices[i] = i;
    }
    
    // Sort indices by fitness (bubble sort for simplicity)
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (opt->population[sort_indices[j]].fitness > opt->population[sort_indices[j + 1]].fitness) {
                int temp = sort_indices[j];
                sort_indices[j] = sort_indices[j + 1];
                sort_indices[j + 1] = temp;
            }
        }
    }
    
    // Assign probabilities based on rank
    for (int i = 0; i < opt->population_size; i++) {
        probabilities[sort_indices[i]] = (opt->population_size - i) / (double)opt->population_size;
    }
    
    // Generate random indices for migration
    int* r1 = (int*)malloc(opt->population_size * sizeof(int));
    int* r3 = (int*)malloc(opt->population_size * sizeof(int));
    
    for (int i = 0; i < opt->population_size; i++) {
        int sequence[opt->population_size - 1];
        int seq_len = 0;
        for (int j = 0; j < opt->population_size; j++) {
            if (j != i) {
                sequence[seq_len++] = j;
            }
        }
        
        int temp = rand() % seq_len;
        r1[i] = sequence[temp];
        sequence[temp] = sequence[--seq_len];
        
        temp = rand() % seq_len;
        r3[i] = sequence[temp];
    }
    
    // Update population
    for (int i = 0; i < opt->population_size; i++) {
        if (rand_double(0.0, 1.0) > probabilities[i]) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = opt->population[r1[i]].position[j] +
                                                rand_double(0.0, 1.0) * (opt->best_solution.position[j] - opt->population[i].position[j]) +
                                                rand_double(0.0, 1.0) * (opt->population[r3[i]].position[j] - opt->population[i].position[j]);
            }
        }
    }
    
    free(probabilities);
    free(sort_indices);
    free(r1);
    free(r3);
    
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void AMO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    initialize_population_amo(opt);
    
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness for all individuals
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
        
        neighborhood_learning_phase(opt);
        global_migration_phase(opt);
        
        // Re-evaluate fitness for updated population
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
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
