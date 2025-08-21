#include "CO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 🌊 Generate a random double between min and max
static inline double rand_double_co(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// 🌊 Fill a buffer with random doubles
void init_rand_buffer(double *buffer, int size, double min, double max) {
    for (int i = 0; i < size; i++) {
        buffer[i] = rand_double_co(min, max);
    }
}

// 🐦 Initialize Cuckoo Population
void initialize_cuckoos(Optimizer *opt) {
    double *rand_buffer = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    init_rand_buffer(rand_buffer, opt->population_size * opt->dim, 0.0, 1.0);
    int idx = 0;
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_buffer[idx++] * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
    free(rand_buffer);
}

// 🥚 Lay Eggs for Each Cuckoo
void lay_eggs(Optimizer *opt, double *egg_positions, int *num_eggs, int *total_eggs, double *rand_buffer) {
    *total_eggs = 0;
    for (int i = 0; i < opt->population_size; i++) {
        num_eggs[i] = MIN_EGGS + (int)(rand_buffer[i] * (MAX_EGGS - MIN_EGGS + 1));
        *total_eggs += num_eggs[i];
    }

    int egg_idx = 0;
    int rand_idx = opt->population_size;  // Start after num_eggs random values
    for (int i = 0; i < opt->population_size; i++) {
        double radius_factor = ((double)num_eggs[i] / *total_eggs) * RADIUS_COEFF;
        for (int k = 0; k < num_eggs[i]; k++) {
            double random_scalar = rand_buffer[rand_idx++];
            double angle = k * (2 * PI / num_eggs[i]);
            int sign = (rand_buffer[rand_idx++] < 0.5) ? 1 : -1;

            for (int j = 0; j < opt->dim; j++) {
                double bound_range = opt->bounds[2 * j + 1] - opt->bounds[2 * j];
                double radius = radius_factor * random_scalar * bound_range;
                double adding_value = sign * radius * cos(angle) + radius * sin(angle);
                double new_pos = opt->population[i].position[j] + adding_value;

                // Enforce bounds
                new_pos = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_pos));
                egg_positions[egg_idx * opt->dim + j] = new_pos;
            }
            egg_idx++;
        }
    }
}

// 🏆 Select Best Cuckoos (Using Partial Selection)
void select_best_cuckoos(Optimizer *opt, double *positions, double *fitness, int num_positions) {
    if (num_positions <= MAX_CUCKOOS) {
        for (int i = 0; i < num_positions; i++) {
            memcpy(opt->population[i].position, &positions[i * opt->dim], opt->dim * sizeof(double));
            opt->population[i].fitness = fitness[i];
        }
        opt->population_size = num_positions;
        return;
    }

    // Use a min-heap to find the top MAX_CUCKOOS
    typedef struct {
        double fitness;
        int index;
    } HeapNode;

    HeapNode *heap = (HeapNode *)malloc(MAX_CUCKOOS * sizeof(HeapNode));
    for (int i = 0; i < MAX_CUCKOOS; i++) {
        heap[i].fitness = fitness[i];
        heap[i].index = i;
    }

    // Build min-heap
    for (int i = MAX_CUCKOOS / 2 - 1; i >= 0; i--) {
        int k = i;
        while (k * 2 + 1 < MAX_CUCKOOS) {
            int child = k * 2 + 1;
            if (child + 1 < MAX_CUCKOOS && heap[child + 1].fitness < heap[child].fitness) {
                child++;
            }
            if (heap[k].fitness <= heap[child].fitness) break;
            HeapNode temp = heap[k];
            heap[k] = heap[child];
            heap[child] = temp;
            k = child;
        }
    }

    // Process remaining elements
    for (int i = MAX_CUCKOOS; i < num_positions; i++) {
        if (fitness[i] < heap[0].fitness) {
            heap[0].fitness = fitness[i];
            heap[0].index = i;
            int k = 0;
            while (k * 2 + 1 < MAX_CUCKOOS) {
                int child = k * 2 + 1;
                if (child + 1 < MAX_CUCKOOS && heap[child + 1].fitness < heap[child].fitness) {
                    child++;
                }
                if (heap[k].fitness <= heap[child].fitness) break;
                HeapNode temp = heap[k];
                heap[k] = heap[child];
                heap[child] = temp;
                k = child;
            }
        }
    }

    // Copy top MAX_CUCKOOS to population
    for (int i = 0; i < MAX_CUCKOOS; i++) {
        int idx = heap[i].index;
        memcpy(opt->population[i].position, &positions[idx * opt->dim], opt->dim * sizeof(double));
        opt->population[i].fitness = fitness[idx];
    }
    opt->population_size = MAX_CUCKOOS;

    free(heap);
}

// 🌐 Cluster and Migrate (Optimized Variance Calculation)
int cluster_and_migrate(Optimizer *opt) {
    // Compute variance efficiently
    double *means = (double *)calloc(opt->dim, sizeof(double));
    double var_sum = 0.0;

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            means[j] += opt->population[i].position[j];
        }
    }
    for (int j = 0; j < opt->dim; j++) {
        means[j] /= opt->population_size;
    }

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double diff = opt->population[i].position[j] - means[j];
            var_sum += diff * diff;
        }
    }
    var_sum /= opt->population_size * opt->dim;
    free(means);

    if (var_sum < VARIANCE_THRESHOLD) {
        return 1;  // Stop optimization
    }

    // Find best cuckoo
    int best_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[best_idx].fitness) {
            best_idx = i;
        }
    }

    // Migrate toward best cuckoo
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double delta = opt->population[best_idx].position[j] - opt->population[i].position[j];
            opt->population[i].position[j] += MOTION_COEFF * rand_double_co(0, 1) * delta;
            opt->population[i].position[j] = fmax(opt->bounds[2 * j], 
                                            fmin(opt->bounds[2 * j + 1], opt->population[i].position[j]));
        }
    }
    enforce_bound_constraints(opt);
    return 0;  // Continue optimization
}

// 🚀 Main Optimization Function
void CO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand(time(NULL));  // Seed random number generator
    initialize_cuckoos(opt);

    // Pre-allocate memory
    int max_total_eggs = opt->population_size * MAX_EGGS_PER_CUCKOO;
    double *egg_positions = (double *)malloc(max_total_eggs * opt->dim * sizeof(double));
    double *egg_fitness = (double *)malloc(max_total_eggs * sizeof(double));
    int *num_eggs = (int *)malloc(opt->population_size * sizeof(int));
    double *all_positions = (double *)malloc((opt->population_size + max_total_eggs) * opt->dim * sizeof(double));
    double *all_fitness = (double *)malloc((opt->population_size + max_total_eggs) * sizeof(double));
    double *rand_buffer = (double *)malloc((opt->population_size + max_total_eggs * 2) * sizeof(double));

    // Evaluate initial population
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
        }
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Generate random numbers for this iteration
        init_rand_buffer(rand_buffer, opt->population_size + max_total_eggs * 2, 0.0, 1.0);

        // Lay eggs
        int total_eggs = 0;
        lay_eggs(opt, egg_positions, num_eggs, &total_eggs, rand_buffer);

        // Evaluate eggs
        for (int i = 0; i < total_eggs; i++) {
            egg_fitness[i] = objective_function(&egg_positions[i * opt->dim]);
        }

        // Combine cuckoos and eggs
        int total_positions = opt->population_size + total_eggs;
        memcpy(all_positions, opt->population[0].position, opt->population_size * opt->dim * sizeof(double));
        memcpy(&all_positions[opt->population_size * opt->dim], egg_positions, total_eggs * opt->dim * sizeof(double));
        for (int i = 0; i < opt->population_size; i++) {
            all_fitness[i] = opt->population[i].fitness;
        }
        memcpy(&all_fitness[opt->population_size], egg_fitness, total_eggs * sizeof(double));

        // Select best cuckoos
        select_best_cuckoos(opt, all_positions, all_fitness, total_positions);

        // Cluster and migrate
        int stop = cluster_and_migrate(opt);

        // Update best solution
        int worst_idx = 0, second_worst_idx = (opt->population_size > 1) ? 1 : 0;
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            }
            if (opt->population[i].fitness > opt->population[worst_idx].fitness) {
                second_worst_idx = worst_idx;
                worst_idx = i;
            } else if (i != worst_idx && opt->population[i].fitness > opt->population[second_worst_idx].fitness) {
                second_worst_idx = i;
            }
        }

        // Replace worst cuckoo with best solution
        if (opt->population[worst_idx].fitness > opt->best_solution.fitness) {
            memcpy(opt->population[worst_idx].position, opt->best_solution.position, opt->dim * sizeof(double));
            opt->population[worst_idx].fitness = opt->best_solution.fitness;
        }

        // Replace second-worst with randomized best
        if (opt->population_size > 1) {
            for (int j = 0; j < opt->dim; j++) {
                double new_pos = opt->best_solution.position[j] * rand_buffer[j];
                new_pos = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_pos));
                opt->population[second_worst_idx].position[j] = new_pos;
            }
            opt->population[second_worst_idx].fitness = objective_function(opt->population[second_worst_idx].position);
        }

        if (stop) {
            break;
        }

        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Clean up
    free(egg_positions);
    free(egg_fitness);
    free(num_eggs);
    free(all_positions);
    free(all_fitness);
    free(rand_buffer);
}
