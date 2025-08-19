#include "CSS.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>

// Fast Xorshift random number generator state
static unsigned int xorshift_state = 1;

// Initialize Xorshift seed
void init_xorshift_css(unsigned int seed) {
    xorshift_state = seed ? seed : 1;
}

// Fast Xorshift random number generator
static inline unsigned int xorshift() {
    unsigned int x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

// Generate random double between min and max
static inline double rand_double_css(double min, double max) {
    return min + (max - min) * ((double)xorshift() / (double)0xFFFFFFFF);
}

// Initialize charged particles
void initialize_charged_particles(Optimizer *opt) {
    register int i, j;
    for (i = 0; i < opt->population_size; i++) {
        for (j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double_css(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Calculate charges and resultant forces with symmetry
void calculate_forces(Optimizer *opt, double *forces) {
    static double *charges = NULL;
    static double *fitness_cache = NULL;
    if (!charges) {
        charges = (double *)malloc(opt->population_size * sizeof(double));
        fitness_cache = (double *)malloc(opt->population_size * sizeof(double));
    }
    double fitworst = -INFINITY, fitbest = INFINITY;
    register int i;
    // Cache fitness values
    for (i = 0; i < opt->population_size; i++) {
        fitness_cache[i] = opt->population[i].fitness;
        if (fitness_cache[i] > fitworst) fitworst = fitness_cache[i];
        if (fitness_cache[i] < fitbest) fitbest = fitness_cache[i];
    }
    // Compute charges
    if (fitbest == fitworst) {
        for (i = 0; i < opt->population_size; i++) {
            charges[i] = 1.0;
        }
    } else {
        for (i = 0; i < opt->population_size; i++) {
            charges[i] = (fitness_cache[i] - fitworst) / (fitbest - fitworst);
        }
    }
    // Initialize forces
    memset(forces, 0, opt->population_size * opt->dim * sizeof(double));
    // Calculate forces with merged loop
    register int j, k;
    for (i = 0; i < opt->population_size; i++) {
        for (j = i + 1; j < opt->population_size; j++) {
            double r_ij = 0.0, r_ij_norm = 0.0;
            double diffs[opt->dim];
            for (k = 0; k < opt->dim; k++) {
                diffs[k] = opt->population[i].position[k] - opt->population[j].position[k];
                r_ij += diffs[k] * diffs[k];
                double norm_diff = (opt->population[i].position[k] + opt->population[j].position[k]) / 2.0 - 
                                  opt->best_solution.position[k];
                r_ij_norm += norm_diff * norm_diff;
            }
            r_ij = sqrt(r_ij);
            r_ij_norm = r_ij / (sqrt(r_ij_norm) + EPSILON);
            double p_ij = (fitness_cache[i] < fitness_cache[j] ||
                          ((fitness_cache[i] - fitbest) / (fitworst - fitbest + EPSILON) > rand_double_css(0.0, 1.0))) ? 1.0 : 0.0;
            double force_term = (r_ij < A) ? (charges[i] * charges[j] / (A * A * A)) * r_ij : (charges[i] * charges[j] / (r_ij * r_ij));
            int i_base = i * opt->dim, j_base = j * opt->dim;
            for (k = 0; k < opt->dim; k++) {
                double force = p_ij * force_term * diffs[k];
                forces[i_base + k] += force;
                forces[j_base + k] -= force;
            }
        }
    }
}

// Update positions and velocities
void css_update_positions(Optimizer *opt, double *forces) {
    static double *velocities = NULL;
    if (!velocities) {
        velocities = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
        memset(velocities, 0, opt->population_size * opt->dim * sizeof(double));
    }
    double dt = 1.0;
    register int i, j;
    for (i = 0; i < opt->population_size; i++) {
        double rand1 = rand_double_css(0.0, 1.0);
        double rand2 = rand_double_css(0.0, 1.0);
        int base_idx = i * opt->dim;
        for (j = 0; j < opt->dim; j++) {
            int idx = base_idx + j;
            velocities[idx] = rand1 * KV * velocities[idx] + rand2 * KA * forces[idx];
            opt->population[i].position[j] += velocities[idx] * dt;
        }
        // Batch boundary checks
        for (j = 0; j < opt->dim; j++) {
            if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                opt->population[i].position[j] = opt->bounds[2 * j];
            } else if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                opt->population[i].position[j] = opt->bounds[2 * j + 1];
            }
        }
    }
}

// Update charged memory with best solutions
void update_charged_memory(Optimizer *opt, double *forces) {
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
    // Partial selection sort for top cm_size
    static int *indices = NULL;
    if (!indices) {
        indices = (int *)malloc(opt->population_size * sizeof(int));
    }
    register int i, j;
    for (i = 0; i < opt->population_size; i++) indices[i] = i;
    for (i = 0; i < cm_size && i < opt->population_size; i++) {
        int min_idx = i;
        for (j = i + 1; j < opt->population_size; j++) {
            if (opt->population[indices[j]].fitness < opt->population[indices[min_idx]].fitness) {
                min_idx = j;
            }
        }
        int temp = indices[i];
        indices[i] = indices[min_idx];
        indices[min_idx] = temp;
    }
    // Update memory
    for (i = 0; i < cm_size && i < opt->population_size; i++) {
        if (memory_size < cm_size) {
            memcpy(charged_memory[memory_size].position, opt->population[indices[i]].position, opt->dim * sizeof(double));
            charged_memory[memory_size].fitness = opt->population[indices[i]].fitness;
            memory_size++;
        } else {
            int worst_idx = 0;
            double worst_fitness = charged_memory[0].fitness;
            for (j = 1; j < memory_size; j++) {
                if (charged_memory[j].fitness > worst_fitness) {
                    worst_fitness = charged_memory[j].fitness;
                    worst_idx = j;
                }
            }
            if (opt->population[indices[i]].fitness < worst_fitness) {
                memcpy(charged_memory[worst_idx].position, opt->population[indices[i]].position, opt->dim * sizeof(double));
                charged_memory[worst_idx].fitness = opt->population[indices[i]].fitness;
            }
        }
    }
}

// Main optimization function
void CSS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    init_xorshift_css((unsigned int)time(NULL));
    initialize_charged_particles(opt);
    double *forces = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    register int iter, i, j;
    for (iter = 0; iter < opt->max_iter; iter++) {
        for (i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            }
        }
        calculate_forces(opt, forces);
        css_update_positions(opt, forces);
        update_charged_memory(opt, forces);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
    free(forces);
}
