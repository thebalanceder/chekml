#include "EFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator
#include <string.h>  // For memcpy()

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Swap two solutions with deep copy
static void swap_solutions(Solution *a, Solution *b, double *temp_position, int dim) {
    memcpy(temp_position, a->position, dim * sizeof(double));
    memcpy(a->position, b->position, dim * sizeof(double));
    memcpy(b->position, temp_position, dim * sizeof(double));
    double temp_fitness = a->fitness;
    a->fitness = b->fitness;
    b->fitness = temp_fitness;
}

// Quicksort partition function
static int partition(Solution *population, int low, int high, double *temp_position, int dim) {
    double pivot = population[high].fitness;
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (population[j].fitness <= pivot) {
            i++;
            swap_solutions(&population[i], &population[j], temp_position, dim);
        }
    }
    swap_solutions(&population[i + 1], &population[high], temp_position, dim);
    return i + 1;
}

// Quicksort implementation
static void quicksort(Solution *population, int low, int high, double *temp_position, int dim) {
    if (low < high) {
        int pi = partition(population, low, high, temp_position, dim);
        quicksort(population, low, pi - 1, temp_position, dim);
        quicksort(population, pi + 1, high, temp_position, dim);
    }
}

// Initialize Electromagnetic Population
void initialize_em_population(Optimizer *opt, ObjectiveFunction obj_func) {
    for (int i = 0; i < opt->population_size; i++) {
        // Use pre-allocated position array (set by general_init)
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = obj_func(opt->population[i].position);
    }
    // Initialize best_solution.position (pre-allocated)
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = 0.0;
    }
    opt->best_solution.fitness = INFINITY;
    // Sort population initially
    double *temp_position = (double *)malloc(opt->dim * sizeof(double));
    if (!temp_position) {
        fprintf(stderr, "Memory allocation failed for temp_position\n");
        exit(1);
    }
    quicksort(opt->population, 0, opt->population_size - 1, temp_position, opt->dim);
    free(temp_position);
    // Update best solution
    if (opt->population[0].fitness < opt->best_solution.fitness) {
        opt->best_solution.fitness = opt->population[0].fitness;
        memcpy(opt->best_solution.position, opt->population[0].position, opt->dim * sizeof(double));
    }
}

// Evaluate and Sort Population
void evaluate_and_sort_population_efo(Optimizer *opt, ObjectiveFunction obj_func, int new_particle_index) {
    // Only evaluate fitness for the new particle (if inserted)
    if (new_particle_index >= 0) {
        opt->population[new_particle_index].fitness = obj_func(opt->population[new_particle_index].position);
    }
    // Sort population using quicksort
    double *temp_position = (double *)malloc(opt->dim * sizeof(double));
    if (!temp_position) {
        fprintf(stderr, "Memory allocation failed for temp_position\n");
        exit(1);
    }
    quicksort(opt->population, 0, opt->population_size - 1, temp_position, opt->dim);
    free(temp_position);
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

// Binary search to find insertion position
static int binary_search_insertion(Solution *population, int size, double new_fitness) {
    int low = 0, high = size - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (population[mid].fitness < new_fitness) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return low;
}

// Insert Particle into Population
void insert_particle_efo(Optimizer *opt, double *new_particle, ObjectiveFunction obj_func, int *new_particle_index) {
    double new_fitness = obj_func(new_particle);
    if (new_fitness < opt->population[opt->population_size - 1].fitness) {
        // Find insertion position using binary search
        int insert_pos = binary_search_insertion(opt->population, opt->population_size, new_fitness);
        // Shift population with deep copy
        for (int i = opt->population_size - 1; i > insert_pos; i--) {
            memcpy(opt->population[i].position, opt->population[i - 1].position, opt->dim * sizeof(double));
            opt->population[i].fitness = opt->population[i - 1].fitness;
        }
        // Insert new particle
        memcpy(opt->population[insert_pos].position, new_particle, opt->dim * sizeof(double));
        opt->population[insert_pos].fitness = new_fitness;
        *new_particle_index = insert_pos;
    } else {
        *new_particle_index = -1;  // Indicate no insertion
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
        int new_particle_index = -1;
        generate_new_particle(opt, iter, new_particle);
        insert_particle_efo(opt, new_particle, obj_func, &new_particle_index);
        evaluate_and_sort_population_efo(opt, obj_func, new_particle_index);

        // Print progress periodically
        if (iter % 1000 == 0) {
            printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
        }
    }

    free(new_particle);
}
