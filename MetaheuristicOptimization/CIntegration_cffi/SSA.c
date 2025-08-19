#include "SSA.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize squirrel positions and context
void initialize_squirrels(Optimizer *opt, ObjectiveFunction objective_function, SSAContext *ctx) {
    int i, j;
    double *bounds = opt->bounds;
    Solution *population = opt->population;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    // Initialize squirrels and tree types
    for (i = 0; i < pop_size; i++) {
        double *pos = population[i].position;
        for (j = 0; j < dim; j++) {
            pos[j] = bounds[2 * j] + rand_double(0.0, 1.0) * (bounds[2 * j + 1] - bounds[2 * j]);
        }
        population[i].fitness = objective_function(pos);
        ctx->tree_types[i] = (rand() % 3) + 1; // 1: acorn, 2: normal, 3: hickory
        ctx->pulse_flying_rates[i] = rand_double(0.0, 1.0); // Precompute pulse flying rate
    }
    enforce_bound_constraints(opt);

    // Find initial best solution
    int min_idx = 0;
    for (i = 1; i < pop_size; i++) {
        if (population[i].fitness < population[min_idx].fitness) {
            min_idx = i;
        }
    }
    opt->best_solution.fitness = population[min_idx].fitness;
    for (j = 0; j < dim; j++) {
        opt->best_solution.position[j] = population[min_idx].position[j];
    }
}

// Update all squirrel positions in a single pass
void update_squirrels(Optimizer *opt, ObjectiveFunction objective_function, SSAContext *ctx) {
    int i, j;
    int dim = opt->dim;
    int pop_size = opt->population_size;
    double *bounds = opt->bounds;
    Solution *population = opt->population;
    double *best_pos = opt->best_solution.position;
    double *velocities = ctx->velocities;
    int *tree_types = ctx->tree_types;
    double *pulse_flying_rates = ctx->pulse_flying_rates;
    double gliding_diff = MAX_GLIDING_DISTANCE - MIN_GLIDING_DISTANCE;

    // Update all squirrels
    for (i = 0; i < pop_size; i++) {
        double *pos = population[i].position;
        double *vel = velocities + i * dim; // Velocity for squirrel i
        double gliding_distance = MIN_GLIDING_DISTANCE + gliding_diff * rand_double(0.0, 1.0);
        int tree_type = tree_types[i];
        double multiplier = (tree_type == 1) ? 1.0 : (tree_type == 2) ? 2.0 : 3.0;

        // Update velocity
        for (j = 0; j < dim; j++) {
            vel[j] += gliding_distance * GLIDING_CONSTANT * (pos[j] - best_pos[j]) * multiplier;
            pos[j] += vel[j];
        }
    }
    enforce_bound_constraints(opt);

    // Random flying condition
    for (i = 0; i < pop_size; i++) {
        if (rand_double(0.0, 1.0) > pulse_flying_rates[i]) {
            double eps = -1.0 + 2.0 * rand_double(0.0, 1.0);
            double mean_A = 0.0;
            for (j = 0; j < pop_size; j++) {
                mean_A += rand_double(0.0, 1.0);
            }
            mean_A /= pop_size;
            double *pos = population[i].position;
            for (j = 0; j < dim; j++) {
                pos[j] = best_pos[j] + eps * mean_A;
            }
        }
    }
    enforce_bound_constraints(opt);

    // Evaluate fitness and update best solution
    for (i = 0; i < pop_size; i++) {
        double *pos = population[i].position;
        double new_fitness = objective_function(pos);
        population[i].fitness = new_fitness;
        if (new_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = new_fitness;
            for (j = 0; j < dim; j++) {
                opt->best_solution.position[j] = pos[j];
            }
        }
    }
}

// Main Optimization Function
void SSA_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    int iter;
    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Seed random number generator
    srand(time(NULL));

    // Allocate SSA context
    SSAContext ctx;
    ctx.tree_types = (int *)malloc(pop_size * sizeof(int));
    ctx.velocities = (double *)calloc(pop_size * dim, sizeof(double));
    ctx.pulse_flying_rates = (double *)malloc(pop_size * sizeof(double));
    if (!ctx.tree_types || !ctx.velocities || !ctx.pulse_flying_rates) {
        fprintf(stderr, "Memory allocation failed for SSA context\n");
        free(ctx.tree_types);
        free(ctx.velocities);
        free(ctx.pulse_flying_rates);
        return;
    }

    // Initialize squirrels
    initialize_squirrels(opt, objective_function, &ctx);

    // Main optimization loop
    for (iter = 0; iter < opt->max_iter; iter++) {
        update_squirrels(opt, objective_function, &ctx);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Clean up
    free(ctx.tree_types);
    free(ctx.velocities);
    free(ctx.pulse_flying_rates);
}
