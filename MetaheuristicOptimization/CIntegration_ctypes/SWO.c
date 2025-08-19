#include "SWO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Function to generate a random double between min and max
double rand_double_swo(double min, double max) {
    double r = (double)rand() / RAND_MAX;
    if (r < 0.0 || r > 1.0) {
        fprintf(stderr, "rand_double_swo: Invalid random value %f\n", r);
        return min;
    }
    return min + (max - min) * r;
}

// Approximate gamma function for Levy flight (simplified for beta = 1.5)
double approximate_gamma(double x) {
    if (x == 1.0) return 1.0;
    if (x == 0.5) return sqrt(M_PI);
    return (x - 1) * approximate_gamma(x - 1);
}

// Generate Levy flight step
double levy_flight_swo() {
    fprintf(stderr, "levy_flight_swo: Starting\n");
    double beta = LEVY_BETA; // Typically 1.5
    // Precomputed sigma for beta = 1.5 to avoid gamma calculation
    double sigma = 0.696066; // Approximate value for beta = 1.5
    fprintf(stderr, "levy_flight_swo: beta=%f, sigma=%f\n", beta, sigma);

    double u = rand_double_swo(-1.0, 1.0) * sigma;
    fprintf(stderr, "levy_flight_swo: u=%f\n", u);
    double v = rand_double_swo(-1.0, 1.0);
    fprintf(stderr, "levy_flight_swo: v=%f\n", v);

    if (v == 0.0) {
        fprintf(stderr, "levy_flight_swo: Division by zero in v\n");
        return 0.0;
    }

    double denom = pow(fabs(v), 1.0 / beta);
    if (denom == 0.0 || !isfinite(denom)) {
        fprintf(stderr, "levy_flight_swo: Invalid denominator %f\n", denom);
        return 0.0;
    }

    double step = u / denom;
    fprintf(stderr, "levy_flight_swo: step=%f\n", step);

    // Cap step to prevent extreme values
    if (step > 1.0 || step < -1.0 || !isfinite(step)) {
        step = (step > 0 || step == 0) ? 1.0 : -1.0;
        fprintf(stderr, "levy_flight_swo: Capped step to %f\n", step);
    }

    double result = LEVY_SCALE * step;
    fprintf(stderr, "levy_flight_swo: Returning %f\n", result);
    return result;
}

// Hunting Phase
void SWO_hunting_phase(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || !opt->population || !opt->best_solution.position || !opt->bounds) {
        fprintf(stderr, "SWO_hunting_phase: Invalid input\n");
        return;
    }

    fprintf(stderr, "SWO_hunting_phase: Starting, population_size=%d, dim=%d\n", opt->population_size, opt->dim);

    // Log bounds
    for (int j = 0; j < 2 * opt->dim; j++) {
        fprintf(stderr, "SWO_hunting_phase: bounds[%d]=%f\n", j, opt->bounds[j]);
    }

    double *new_pos = (double *)malloc(opt->dim * sizeof(double));
    if (!new_pos) {
        fprintf(stderr, "SWO_hunting_phase: Failed to allocate new_pos\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "SWO_hunting_phase: Null position for population[%d]\n", i);
            free(new_pos);
            return;
        }

        // Log all position values
        fprintf(stderr, "SWO_hunting_phase: Processing individual %d, fitness=%f\n", i, opt->population[i].fitness);
        for (int j = 0; j < opt->dim; j++) {
            fprintf(stderr, "SWO_hunting_phase: Individual %d, position[%d]=%f, best_position[%d]=%f\n", 
                    i, j, opt->population[i].position[j], j, opt->best_solution.position[j]);
        }

        fprintf(stderr, "SWO_hunting_phase: Individual %d, generating random values\n", i);
        double r1 = rand_double_swo(0.0, 1.0);
        fprintf(stderr, "SWO_hunting_phase: Individual %d, r1=%f\n", i, r1);
        double r2 = rand_double_swo(0.0, 1.0);
        fprintf(stderr, "SWO_hunting_phase: Individual %d, r2=%f\n", i, r2);
        double L = levy_flight_swo();
        fprintf(stderr, "SWO_hunting_phase: Individual %d, L=%f\n", i, L);

        for (int j = 0; j < opt->dim; j++) {
            if (2 * j + 1 >= 2 * opt->dim) {
                fprintf(stderr, "SWO_hunting_phase: Invalid bounds access at j=%d, dim=%d\n", j, opt->dim);
                free(new_pos);
                return;
            }

            double bound_min = opt->bounds[2 * j];
            double bound_max = opt->bounds[2 * j + 1];
            if (bound_min >= bound_max) {
                fprintf(stderr, "SWO_hunting_phase: Invalid bounds for dim %d, min=%f, max=%f\n", 
                        j, bound_min, bound_max);
                free(new_pos);
                return;
            }

            fprintf(stderr, "SWO_hunting_phase: Individual %d, dim %d, bounds=[%f, %f]\n", 
                    i, j, bound_min, bound_max);

            if (r1 < TRADE_OFF) {
                // Exploration: Move towards best solution with Levy flight
                double delta = opt->best_solution.position[j] - opt->population[i].position[j];
                new_pos[j] = opt->population[i].position[j] + L * delta;
            } else {
                // Exploitation: Local search around current position
                double bound_range = bound_max - bound_min;
                new_pos[j] = opt->population[i].position[j] + r2 * bound_range * rand_double_swo(-0.1, 0.1);
            }
            // Bound checking
            new_pos[j] = fmax(bound_min, fmin(bound_max, new_pos[j]));
            fprintf(stderr, "SWO_hunting_phase: Individual %d, dim %d, new_pos=%f\n", i, j, new_pos[j]);
        }

        // Evaluate new position
        fprintf(stderr, "SWO_hunting_phase: Individual %d, evaluating objective function\n", i);
        double new_fitness = objective_function(new_pos);
        fprintf(stderr, "SWO_hunting_phase: Individual %d, new_fitness=%f, old_fitness=%f\n", 
                i, new_fitness, opt->population[i].fitness);

        if (new_fitness < opt->population[i].fitness) {
            opt->population[i].fitness = new_fitness;
            memcpy(opt->population[i].position, new_pos, opt->dim * sizeof(double));
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                memcpy(opt->best_solution.position, new_pos, opt->dim * sizeof(double));
            }
        }
    }

    free(new_pos);
    enforce_bound_constraints(opt);
    fprintf(stderr, "SWO_hunting_phase: Completed\n");
}

// Mating Phase
void SWO_mating_phase(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || !opt->population || !opt->bounds) {
        fprintf(stderr, "SWO_mating_phase: Invalid input\n");
        return;
    }

    fprintf(stderr, "SWO_mating_phase: Starting\n");

    double *new_pos = (double *)malloc(opt->dim * sizeof(double));
    if (!new_pos) {
        fprintf(stderr, "SWO_mating_phase: Failed to allocate new_pos\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "SWO_mating_phase: Null position for population[%d]\n", i);
            free(new_pos);
            return;
        }

        // Select a random mate
        int mate_idx = rand() % opt->population_size;
        if (mate_idx == i || mate_idx >= opt->population_size || !opt->population[mate_idx].position) {
            continue;
        }

        fprintf(stderr, "SWO_mating_phase: Individual %d, mate_idx=%d\n", i, mate_idx);

        for (int j = 0; j < opt->dim; j++) {
            if (2 * j + 1 >= 2 * opt->dim) {
                fprintf(stderr, "SWO_mating_phase: Invalid bounds access at j=%d, dim=%d\n", j, opt->dim);
                free(new_pos);
                return;
            }

            if (rand_double_swo(0.0, 1.0) < CROSSOVER_PROB) {
                // Crossover with mate
                new_pos[j] = opt->population[i].position[j] + 
                             rand_double_swo(0.0, 1.0) * (opt->population[mate_idx].position[j] - opt->population[i].position[j]);
            } else {
                new_pos[j] = opt->population[i].position[j];
            }
            // Bound checking
            new_pos[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_pos[j]));
        }

        // Evaluate new position
        double new_fitness = objective_function(new_pos);
        if (new_fitness < opt->population[i].fitness) {
            opt->population[i].fitness = new_fitness;
            memcpy(opt->population[i].position, new_pos, opt->dim * sizeof(double));
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                memcpy(opt->best_solution.position, new_pos, opt->dim * sizeof(double));
            }
        }
    }

    free(new_pos);
    enforce_bound_constraints(opt);
    fprintf(stderr, "SWO_mating_phase: Completed\n");
}

// Population Reduction
void SWO_population_reduction(Optimizer *opt, int iter) {
    if (!opt || !opt->population || !opt->population[0].position) {
        fprintf(stderr, "SWO_population_reduction: Invalid optimizer\n");
        return;
    }

    fprintf(stderr, "SWO_population_reduction: Starting, iter=%d\n", iter);

    double T = (double)opt->max_iter;
    int current_population = MIN_POPULATION + (int)((opt->population_size - MIN_POPULATION) * ((T - iter) / T));
    if (current_population < MIN_POPULATION) {
        current_population = MIN_POPULATION;
    }

    if (current_population < opt->population_size) {
        // Sort population by fitness
        for (int i = 0; i < opt->population_size - 1; i++) {
            for (int j = 0; j < opt->population_size - i - 1; j++) {
                if (!opt->population[j].position || !opt->population[j + 1].position) {
                    fprintf(stderr, "SWO_population_reduction: Null position at index %d or %d\n", j, j + 1);
                    return;
                }
                if (opt->population[j].fitness > opt->population[j + 1].fitness) {
                    // Swap fitness and position data
                    double fitness_temp = opt->population[j].fitness;
                    opt->population[j].fitness = opt->population[j + 1].fitness;
                    opt->population[j + 1].fitness = fitness_temp;

                    double *temp_pos = (double *)malloc(opt->dim * sizeof(double));
                    if (!temp_pos) {
                        fprintf(stderr, "SWO_population_reduction: Failed to allocate temp_pos\n");
                        return;
                    }
                    memcpy(temp_pos, opt->population[j].position, opt->dim * sizeof(double));
                    memcpy(opt->population[j].position, opt->population[j + 1].position, opt->dim * sizeof(double));
                    memcpy(opt->population[j + 1].position, temp_pos, opt->dim * sizeof(double));
                    free(temp_pos);
                }
            }
        }

        // Update population size
        int old_population_size = opt->population_size;
        opt->population_size = current_population;

        // Reallocate population structure
        Solution *new_population = (Solution *)realloc(opt->population, opt->population_size * sizeof(Solution));
        if (!new_population) {
            fprintf(stderr, "SWO_population_reduction: Failed to realloc population\n");
            opt->population_size = old_population_size;
            return;
        }
        opt->population = new_population;

        // Allocate new position memory
        double *new_positions = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
        if (!new_positions) {
            fprintf(stderr, "SWO_population_reduction: Failed to allocate new_positions\n");
            opt->population_size = old_population_size;
            return;
        }

        // Copy position data to new memory
        for (int i = 0; i < opt->population_size; i++) {
            if (!opt->population[i].position) {
                fprintf(stderr, "SWO_population_reduction: Null position for population[%d]\n", i);
                free(new_positions);
                opt->population_size = old_population_size;
                return;
            }
            memcpy(new_positions + (i * opt->dim), opt->population[i].position, opt->dim * sizeof(double));
        }

        // Free old position memory and update pointers
        if (opt->population[0].position) {
            free(opt->population[0].position);
        }
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].position = new_positions + (i * opt->dim);
        }
    }

    enforce_bound_constraints(opt);
    fprintf(stderr, "SWO_population_reduction: Completed, new population size=%d\n", opt->population_size);
}

// Main Optimization Function
void SWO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || !opt->population || !opt->best_solution.position) {
        fprintf(stderr, "SWO_optimize: Invalid input\n");
        return;
    }

    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Initialize population fitness
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "SWO_optimize: Null position for population[%d]\n", i);
            return;
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    // Log initial population state
    fprintf(stderr, "SWO_optimize: Initial population state\n");
    for (int i = 0; i < opt->population_size; i++) {
        fprintf(stderr, "SWO_optimize: Individual %d, fitness=%f, position[0]=%f\n", 
                i, opt->population[i].fitness, opt->population[i].position[0]);
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        fprintf(stderr, "SWO_optimize: Iteration %d, Population size: %d\n", iter, opt->population_size);
        if (rand_double_swo(0.0, 1.0) < TRADE_OFF) {
            SWO_hunting_phase(opt, objective_function);
        } else {
            SWO_mating_phase(opt, objective_function);
        }
        SWO_population_reduction(opt, iter);
    }
}
