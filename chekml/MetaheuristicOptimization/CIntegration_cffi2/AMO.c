#include "AMO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Inline utility functions
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

static inline double normal_rand(double mean, double stddev) {
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

// Quicksort implementation for sorting indices by fitness
static void quicksort_indices(Optimizer *opt, int *indices, int low, int high) {
    if (low < high) {
        double pivot = opt->population[indices[high]].fitness;
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            if (opt->population[indices[j]].fitness <= pivot) {
                i++;
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
        
        int temp = indices[i + 1];
        indices[i + 1] = indices[high];
        indices[high] = temp;
        
        quicksort_indices(opt, indices, low, i);
        quicksort_indices(opt, indices, i + 2, high);
    }
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
    double *new_pop = (double*)malloc(opt->population_size * opt->dim * sizeof(double));
    
    for (int i = 0; i < opt->population_size; i++) {
        double FF = normal_rand(0.0, 1.0);
        int lseq[NEIGHBORHOOD_SIZE];
        
        // Define neighborhood
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
        
        // Random permutation
        for (int j = 0; j < NEIGHBORHOOD_SIZE; j++) {
            int temp = lseq[j];
            int idx = rand() % NEIGHBORHOOD_SIZE;
            lseq[j] = lseq[idx];
            lseq[idx] = temp;
        }
        
        int exemplar_idx = lseq[1];
        
        // Compute new position
        int offset = i * opt->dim;
        for (int j = 0; j < opt->dim; j++) {
            new_pop[offset + j] = opt->population[i].position[j] + 
                                  FF * (opt->population[exemplar_idx].position[j] - opt->population[i].position[j]);
        }
    }
    
    // Copy back to population
    for (int i = 0; i < opt->population_size; i++) {
        int offset = i * opt->dim;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = new_pop[offset + j];
        }
    }
    
    free(new_pop);
    enforce_bound_constraints(opt);
}

// Global Migration Phase
void global_migration_phase(Optimizer *opt) {
    double *new_pop = (double*)malloc(opt->population_size * opt->dim * sizeof(double));
    double *probabilities = (double*)malloc(opt->population_size * sizeof(double));
    int *sort_indices = (int*)malloc(opt->population_size * sizeof(int));
    int *r1 = (int*)malloc(opt->population_size * sizeof(int));
    int *r3 = (int*)malloc(opt->population_size * sizeof(int));
    
    // Initialize sort_indices
    for (int i = 0; i < opt->population_size; i++) {
        sort_indices[i] = i;
        probabilities[i] = 0.0;
    }
    
    // Sort indices by fitness
    quicksort_indices(opt, sort_indices, 0, opt->population_size - 1);
    
    // Assign probabilities
    for (int i = 0; i < opt->population_size; i++) {
        probabilities[sort_indices[i]] = (opt->population_size - i) / (double)opt->population_size;
    }
    
    // Generate random indices
    for (int i = 0; i < opt->population_size; i++) {
        int indices[opt->population_size];
        for (int j = 0; j < opt->population_size; j++) {
            indices[j] = j;
        }
        
        // Fisher-Yates shuffle
        for (int j = opt->population_size - 1; j > 0; j--) {
            int k = rand() % (j + 1);
            int temp = indices[j];
            indices[j] = indices[k];
            indices[k] = temp;
        }
        
        // Select first two valid indices (excluding i)
        int idx = 0;
        while (indices[idx] == i) idx++;
        r1[i] = indices[idx++];
        while (indices[idx] == i) idx++;
        r3[i] = indices[idx];
    }
    
    // Update population
    for (int i = 0; i < opt->population_size; i++) {
        int offset = i * opt->dim;
        if (rand_double(0.0, 1.0) > probabilities[i]) {
            for (int j = 0; j < opt->dim; j++) {
                new_pop[offset + j] = opt->population[r1[i]].position[j] +
                                      rand_double(0.0, 1.0) * (opt->best_solution.position[j] - opt->population[i].position[j]) +
                                      rand_double(0.0, 1.0) * (opt->population[r3[i]].position[j] - opt->population[i].position[j]);
            }
        } else {
            for (int j = 0; j < opt->dim; j++) {
                new_pop[offset + j] = opt->population[i].position[j];
            }
        }
    }
    
    // Copy back to population
    for (int i = 0; i < opt->population_size; i++) {
        int offset = i * opt->dim;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = new_pop[offset + j];
        }
    }
    
    free(new_pop);
    free(probabilities);
    free(sort_indices);
    free(r1);
    free(r3);
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void AMO_optimize(void *opt_void, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer*)opt_void;
    
    initialize_population_amo(opt);
    
    // Evaluate initial fitness
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
        neighborhood_learning_phase(opt);
        
        // Evaluate fitness for updated population
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            if (new_fitness <= opt->population[i].fitness) {
                opt->population[i].fitness = new_fitness;
                if (new_fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = new_fitness;
                    for (int j = 0; j < opt->dim; j++) {
                        opt->best_solution.position[j] = opt->population[i].position[j];
                    }
                }
            }
        }
        
        global_migration_phase(opt);
        
        // Evaluate fitness for updated population
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            if (new_fitness <= opt->population[i].fitness) {
                opt->population[i].fitness = new_fitness;
                if (new_fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = new_fitness;
                    for (int j = 0; j < opt->dim; j++) {
                        opt->best_solution.position[j] = opt->population[i].position[j];
                    }
                }
            }
        }
        
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
