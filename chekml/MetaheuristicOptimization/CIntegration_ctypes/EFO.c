#include "EFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator
#include <string.h>  // For memcpy()

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize Electromagnetic Population
void initialize_em_population(Optimizer *opt, ObjectiveFunction obj_func) {
    for (int i = 0; i < opt->population_size; i++) {
        // Use pre-allocated position array (set by general_init)
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;  // Will be updated in evaluate_and_sort
    }
    // Initialize best_solution.position (pre-allocated)
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = 0.0;  // Initialize to zero or a valid value
    }
    opt->best_solution.fitness = INFINITY;
    evaluate_and_sort_population(opt, obj_func);
}

// Evaluate and Sort Population
void evaluate_and_sort_population(Optimizer *opt, ObjectiveFunction obj_func) {
    // Compute fitness for each particle
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = obj_func(opt->population[i].position);
    }

    // Sort population by fitness (ascending) with deep copy
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (opt->population[j].fitness > opt->population[j + 1].fitness) {
                // Perform deep copy swap within pre-allocated position arrays
                double *temp_position = (double *)malloc(opt->dim * sizeof(double));
                if (!temp_position) {
                    fprintf(stderr, "Memory allocation failed for temp_position\n");
                    exit(1);
                }
                // Copy position arrays
                memcpy(temp_position, opt->population[j].position, opt->dim * sizeof(double));
                memcpy(opt->population[j].position, opt->population[j + 1].position, opt->dim * sizeof(double));
                memcpy(opt->population[j + 1].position, temp_position, opt->dim * sizeof(double));
                free(temp_position);
                // Swap fitness
                double temp_fitness = opt->population[j].fitness;
                opt->population[j].fitness = opt->population[j + 1].fitness;
                opt->population[j + 1].fitness = temp_fitness;
            }
        }
    }

    // Update best solution
    if (opt->population[0].fitness < opt->best_solution.fitness) {
        opt->best_solution.fitness = opt->population[0].fitness;
        memcpy(opt->best_solution.position, opt->population[0].position, opt->dim * sizeof(double));
    }
}

// Generate New Particle
void generate_new_particle(Optimizer *opt, int t, double *new_particle) {
    double r = rand_double(0.0, 1.0);  // Random force
    double rp = rand_double(0.0, 1.0);  // Randomization probability
    double randomization = rand_double(0.0, 1.0);  // Randomization coefficient

    // Precompute random indices
    int r_index1 = rand() % POSITIVE_FIELD_SIZE(opt->population_size);  // Positive field
    int r_index2 = NEGATIVE_FIELD_START(opt->population_size) + 
                   (rand() % (opt->population_size - NEGATIVE_FIELD_START(opt->population_size)));  // Negative field
    int r_index3 = NEUTRAL_FIELD_START(opt->population_size) + 
                   (rand() % (NEUTRAL_FIELD_END(opt->population_size) - NEUTRAL_FIELD_START(opt->population_size)));  // Neutral field

    for (int i = 0; i < opt->dim; i++) {
        double ps = rand_double(0.0, 1.0);  // Selection probability
        if (ps > POSITIVE_SELECTION_RATE) {
            // Use particles from positive, neutral, and negative fields
            new_particle[i] = (opt->population[r_index3].position[i] +
                              GOLDEN_RATIO * r * (opt->population[r_index1].position[i] - 
                                                 opt->population[r_index3].position[i]) +
                              r * (opt->population[r_index3].position[i] - 
                                   opt->population[r_index2].position[i]));
        } else {
            // Copy from positive field
            new_particle[i] = opt->population[r_index1].position[i];
        }

        // Check boundaries
        if (new_particle[i] < opt->bounds[2 * i] || new_particle[i] > opt->bounds[2 * i + 1]) {
            new_particle[i] = opt->bounds[2 * i] + 
                             (opt->bounds[2 * i + 1] - opt->bounds[2 * i]) * randomization;
        }
    }

    // Randomize one dimension with probability RANDOMIZATION_RATE
    if (rp < RANDOMIZATION_RATE) {
        int ri = rand() % opt->dim;
        new_particle[ri] = opt->bounds[2 * ri] + 
                          (opt->bounds[2 * ri + 1] - opt->bounds[2 * ri]) * randomization;
    }
}

// Insert Particle into Population
void insert_particle(Optimizer *opt, double *new_particle, ObjectiveFunction obj_func) {
    double new_fitness = obj_func(new_particle);
    if (new_fitness < opt->population[opt->population_size - 1].fitness) {
        // Find insertion position
        int insert_pos = 0;
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness <= new_fitness) {
                insert_pos = i + 1;
            } else {
                break;
            }
        }

        // Shift population with deep copy
        for (int i = opt->population_size - 1; i > insert_pos; i--) {
            memcpy(opt->population[i].position, opt->population[i - 1].position, opt->dim * sizeof(double));
            opt->population[i].fitness = opt->population[i - 1].fitness;
        }

        // Insert new particle
        memcpy(opt->population[insert_pos].position, new_particle, opt->dim * sizeof(double));
        opt->population[insert_pos].fitness = new_fitness;
    }
}

// Main Optimization Function
void EFO_optimize(void *opt_void, ObjectiveFunction obj_func) {
    Optimizer *opt = (Optimizer *)opt_void;

    initialize_em_population(opt, obj_func);

    double *new_particle = (double *)malloc(opt->dim * sizeof(double));
    if (!new_particle) {
        fprintf(stderr, "Memory allocation failed for new_particle\n");
        exit(1);
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        generate_new_particle(opt, iter, new_particle);
        insert_particle(opt, new_particle, obj_func);
        evaluate_and_sort_population(opt, obj_func);

        // Print progress periodically
        if (iter % 1000 == 0) {
            printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
        }
    }

    free(new_particle);
}
