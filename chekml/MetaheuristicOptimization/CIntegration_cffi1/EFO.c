#include "EFO.h"
#include "generaloptimizer.h"
#include <string.h>  // For memcpy()
#include <time.h>    // For time()

// Initialize Xorshift state
static inline void xorshift_init(XorshiftState *rng, uint32_t seed) {
    rng->state = seed ? seed : 1;
}

// Generate random uint32_t using Xorshift
static inline uint32_t xorshift_next_efo(XorshiftState *rng) {
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

// Generate random double in [min, max)
static inline double xorshift_double_efo(XorshiftState *rng, double min, double max) {
    return min + (max - min) * (xorshift_next_efo(rng) / (double)UINT32_MAX);
}

// Swap two solutions with deep copy
static inline void swap_solutions(Solution *a, Solution *b, double *temp_position, int dim) {
    memcpy(temp_position, a->position, dim * sizeof(double));
    memcpy(a->position, b->position, dim * sizeof(double));
    memcpy(b->position, temp_position, dim * sizeof(double));
    double temp_fitness = a->fitness;
    a->fitness = b->fitness;
    b->fitness = temp_fitness;
}

// Initialize Electromagnetic Population
void initialize_em_population(Optimizer *opt, ObjectiveFunction obj_func) {
    XorshiftState rng;
    xorshift_init(&rng, (uint32_t)time(NULL));

    for (int i = 0; i < opt->population_size; i++) {
        // Use pre-allocated position array (set by general_init)
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = xorshift_double_efo(&rng, opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = obj_func(opt->population[i].position);
    }
    // Initialize best_solution.position (pre-allocated)
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = 0.0;
    }
    opt->best_solution.fitness = INFINITY;

    // Insertion sort for initial population (small n, cache-friendly)
    double *temp_position = (double *)malloc(opt->dim * sizeof(double));
    if (!temp_position) {
        fprintf(stderr, "Memory allocation failed for temp_position\n");
        exit(1);
    }
    for (int i = 1; i < opt->population_size; i++) {
        Solution key = opt->population[i];
        memcpy(temp_position, key.position, opt->dim * sizeof(double));
        int j = i - 1;
        while (j >= 0 && opt->population[j].fitness > key.fitness) {
            opt->population[j + 1] = opt->population[j];
            j--;
        }
        memcpy(opt->population[j + 1].position, temp_position, opt->dim * sizeof(double));
        opt->population[j + 1].fitness = key.fitness;
    }
    free(temp_position);

    // Update best solution
    if (opt->population[0].fitness < opt->best_solution.fitness) {
        opt->best_solution.fitness = opt->population[0].fitness;
        memcpy(opt->best_solution.position, opt->population[0].position, opt->dim * sizeof(double));
    }
}

// Evaluate and Sort Population (insertion-based for new particle)
void evaluate_and_sort_population(Optimizer *opt, ObjectiveFunction obj_func, int new_particle_index) {
    if (new_particle_index < 0) return;  // No new particle inserted

    // Evaluate fitness for the new particle
    opt->population[new_particle_index].fitness = obj_func(opt->population[new_particle_index].position);

    // Move the new particle to its sorted position
    double *temp_position = (double *)malloc(opt->dim * sizeof(double));
    if (!temp_position) {
        fprintf(stderr, "Memory allocation failed for temp_position\n");
        exit(1);
    }
    Solution key = opt->population[new_particle_index];
    memcpy(temp_position, key.position, opt->dim * sizeof(double));
    int j = new_particle_index - 1;
    while (j >= 0 && opt->population[j].fitness > key.fitness) {
        opt->population[j + 1] = opt->population[j];
        j--;
    }
    memcpy(opt->population[j + 1].position, temp_position, opt->dim * sizeof(double));
    opt->population[j + 1].fitness = key.fitness;
    free(temp_position);

    // Update best solution
    if (opt->population[0].fitness < opt->best_solution.fitness) {
        opt->best_solution.fitness = opt->population[0].fitness;
        memcpy(opt->best_solution.position, opt->population[0].position, opt->dim * sizeof(double));
    }
}

// Generate New Particle
void generate_new_particle(Optimizer *opt, int t, double *new_particle, XorshiftState *rng, double *random_values) {
    // Use pre-generated random values
    double r = random_values[0];  // Random force
    double rp = random_values[1];  // Randomization probability
    double randomization = random_values[2];  // Randomization coefficient
    double ps = random_values[3];  // Selection probability

    // Precompute random indices
    int r_index1 = (int)(xorshift_next_efo(rng) % POSITIVE_FIELD_SIZE(opt->population_size));  // Positive field
    int r_index2 = NEGATIVE_FIELD_START(opt->population_size) + 
                   (xorshift_next_efo(rng) % (opt->population_size - NEGATIVE_FIELD_START(opt->population_size)));  // Negative field
    int r_index3 = NEUTRAL_FIELD_START(opt->population_size) + 
                   (xorshift_next_efo(rng) % (NEUTRAL_FIELD_END(opt->population_size) - NEUTRAL_FIELD_START(opt->population_size)));  // Neutral field

    for (int i = 0; i < opt->dim; i++) {
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
        int ri = (int)(xorshift_next_efo(rng) % opt->dim);
        new_particle[ri] = opt->bounds[2 * ri] + 
                          (opt->bounds[2 * ri + 1] - opt->bounds[2 * ri]) * randomization;
    }
}

// Insert Particle into Population
void insert_particle(Optimizer *opt, double *new_particle, ObjectiveFunction obj_func, int *new_particle_index, double *temp_position) {
    double new_fitness = obj_func(new_particle);
    if (new_fitness < opt->population[opt->population_size - 1].fitness) {
        // Find insertion position (linear search, as population is sorted)
        int insert_pos = opt->population_size - 1;
        while (insert_pos > 0 && opt->population[insert_pos - 1].fitness > new_fitness) {
            insert_pos--;
        }
        // Shift population
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

    // Initialize RNG
    XorshiftState rng;
    xorshift_init(&rng, (uint32_t)time(NULL));

    // Pre-allocate buffers
    double *new_particle = (double *)malloc(opt->dim * sizeof(double));
    double *temp_position = (double *)malloc(opt->dim * sizeof(double));
    double *random_values = (double *)malloc(4 * sizeof(double));  // For r, rp, randomization, ps
    if (!new_particle || !temp_position || !random_values) {
        fprintf(stderr, "Memory allocation failed\n");
        free(new_particle);
        free(temp_position);
        free(random_values);
        exit(1);
    }

    initialize_em_population(opt, obj_func);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        int new_particle_index = -1;

        // Generate random values in bulk
        for (int i = 0; i < 4; i++) {
            random_values[i] = xorshift_double_efo(&rng, 0.0, 1.0);
        }

        generate_new_particle(opt, iter, new_particle, &rng, random_values);
        insert_particle(opt, new_particle, obj_func, &new_particle_index, temp_position);
        evaluate_and_sort_population(opt, obj_func, new_particle_index);

        // Print progress periodically
        if (iter % 1000 == 0) {
            printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
        }
    }

    // Clean up
    free(new_particle);
    free(temp_position);
    free(random_values);
}
