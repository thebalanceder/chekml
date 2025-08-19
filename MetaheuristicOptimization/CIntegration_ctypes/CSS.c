#include "CSS.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize charged particles
void initialize_charged_particles(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Calculate charges and resultant forces
void calculate_forces(Optimizer *opt, double *forces) {
    double *charges = (double *)malloc(opt->population_size * sizeof(double));
    double fitworst = -INFINITY, fitbest = INFINITY;
    // Find best and worst fitness
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness > fitworst) fitworst = opt->population[i].fitness;
        if (opt->population[i].fitness < fitbest) fitbest = opt->population[i].fitness;
    }
    // Compute charges
    if (fitbest == fitworst) {
        for (int i = 0; i < opt->population_size; i++) {
            charges[i] = 1.0;
        }
    } else {
        for (int i = 0; i < opt->population_size; i++) {
            charges[i] = (opt->population[i].fitness - fitworst) / (fitbest - fitworst);
        }
    }
    // Initialize forces
    memset(forces, 0, opt->population_size * opt->dim * sizeof(double));
    // Calculate forces for each particle
    for (int j = 0; j < opt->population_size; j++) {
        for (int i = 0; i < opt->population_size; i++) {
            if (i != j) {
                double r_ij = 0.0, r_ij_norm = 0.0;
                for (int k = 0; k < opt->dim; k++) {
                    double diff = opt->population[i].position[k] - opt->population[j].position[k];
                    r_ij += diff * diff;
                    double norm_diff = (opt->population[i].position[k] + opt->population[j].position[k]) / 2.0 - 
                                      opt->best_solution.position[k];
                    r_ij_norm += norm_diff * norm_diff;
                }
                r_ij = sqrt(r_ij);
                r_ij_norm = r_ij / (sqrt(r_ij_norm) + EPSILON);
                // Probability of attraction
                double p_ij = (opt->population[i].fitness < opt->population[j].fitness ||
                              ((opt->population[i].fitness - fitbest) / (fitworst - fitbest + EPSILON) > rand_double(0.0, 1.0))) ? 1.0 : 0.0;
                // Force calculation
                double force_term = (r_ij < A) ? (charges[i] / (A * A * A)) * r_ij : (charges[i] / (r_ij * r_ij));
                for (int k = 0; k < opt->dim; k++) {
                    forces[j * opt->dim + k] += p_ij * force_term * 
                                               (opt->population[i].position[k] - opt->population[j].position[k]) * charges[j];
                }
            }
        }
    }
    free(charges);
}

// Update positions and velocities
void css_update_positions(Optimizer *opt, double *forces) {
    static double *velocities = NULL;
    if (!velocities) {
        velocities = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
        memset(velocities, 0, opt->population_size * opt->dim * sizeof(double));
    }
    double dt = 1.0;
    for (int i = 0; i < opt->population_size; i++) {
        double rand1 = rand_double(0.0, 1.0);
        double rand2 = rand_double(0.0, 1.0);
        for (int j = 0; j < opt->dim; j++) {
            int idx = i * opt->dim + j;
            velocities[idx] = rand1 * KV * velocities[idx] + rand2 * KA * forces[idx];
            opt->population[i].position[j] += velocities[idx] * dt;
            if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                opt->population[i].position[j] = opt->bounds[2 * j];
            } else if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                opt->population[i].position[j] = opt->bounds[2 * j + 1];
            }
        }
    }
}

// Update charged memory with best solutions
void update_charged_memory(Optimizer *opt, double *charges) {
    int cm_size = (int)(CM_SIZE_RATIO * opt->population_size);
    static Solution *charged_memory = NULL;
    static int memory_size = 0;
    if (!charged_memory) {
        charged_memory = (Solution *)malloc(cm_size * sizeof(Solution));
        for (int i = 0; i < cm_size; i++) {
            charged_memory[i].position = (double *)malloc(opt->dim * sizeof(double));
        }
        memory_size = 0;
    }
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    for (int i = 0; i < opt->population_size; i++) indices[i] = i;
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = i + 1; j < opt->population_size; j++) {
            if (opt->population[indices[i]].fitness > opt->population[indices[j]].fitness) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    for (int i = 0; i < cm_size && i < opt->population_size; i++) {
        if (memory_size < cm_size) {
            for (int j = 0; j < opt->dim; j++) {
                charged_memory[memory_size].position[j] = opt->population[indices[i]].position[j];
            }
            charged_memory[memory_size].fitness = opt->population[indices[i]].fitness;
            memory_size++;
        } else {
            int worst_idx = 0;
            double worst_fitness = charged_memory[0].fitness;
            for (int j = 1; j < memory_size; j++) {
                if (charged_memory[j].fitness > worst_fitness) {
                    worst_fitness = charged_memory[j].fitness;
                    worst_idx = j;
                }
            }
            if (opt->population[indices[i]].fitness < worst_fitness) {
                for (int j = 0; j < opt->dim; j++) {
                    charged_memory[worst_idx].position[j] = opt->population[indices[i]].position[j];
                }
                charged_memory[worst_idx].fitness = opt->population[indices[i]].fitness;
            }
        }
    }
    free(indices);
}

// Main optimization function
void CSS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    initialize_charged_particles(opt);
    double *forces = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    for (int iter = 0; iter < opt->max_iter; iter++) {
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        calculate_forces(opt, forces);
        css_update_positions(opt, forces);
        update_charged_memory(opt, forces);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
    free(forces);
}
