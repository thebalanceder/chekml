#include "SWO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Function to generate a random double between min and max
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Generate Levy flight step (optimized for beta = 1.5)
static inline double levy_flight_swo() {
    static const double sigma = 0.696066; // Precomputed for beta = 1.5
    double u = rand_double(-1.0, 1.0) * sigma;
    double v = rand_double(-1.0, 1.0);
    if (v == 0.0) return 0.0;
    double step = u / pow(fabs(v), 0.666667); // 1 / 1.5 = 0.666667
    if (step > 1.0 || step < -1.0 || !isfinite(step)) {
        return (step >= 0) ? LEVY_SCALE : -LEVY_SCALE;
    }
    return LEVY_SCALE * step;
}

// Hunting Phase (optimized for small dimensions)
void SWO_hunting_phase(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || !opt->population || !opt->best_solution.position || !opt->bounds) return;

    double *new_pos = (double *)malloc(opt->dim * sizeof(double));
    if (!new_pos) return;

    // Cache bounds ranges for efficiency
    double bound_range[opt->dim];
    for (int j = 0; j < opt->dim; j++) {
        bound_range[j] = opt->bounds[2 * j + 1] - opt->bounds[2 * j];
        if (bound_range[j] <= 0) {
            free(new_pos);
            return;
        }
    }

    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            free(new_pos);
            return;
        }

        double r1 = rand_double(0.0, 1.0);
        double r2 = rand_double(0.0, 1.0);
        double L = levy_flight_swo();

        // Unroll loop for dim=2 to reduce branching
        if (opt->dim == 2) {
            if (r1 < TRADE_OFF) {
                new_pos[0] = opt->population[i].position[0] + L * (opt->best_solution.position[0] - opt->population[i].position[0]);
                new_pos[1] = opt->population[i].position[1] + L * (opt->best_solution.position[1] - opt->population[i].position[1]);
            } else {
                new_pos[0] = opt->population[i].position[0] + r2 * bound_range[0] * rand_double(-0.1, 0.1);
                new_pos[1] = opt->population[i].position[1] + r2 * bound_range[1] * rand_double(-0.1, 0.1);
            }
            new_pos[0] = fmax(opt->bounds[0], fmin(opt->bounds[1], new_pos[0]));
            new_pos[1] = fmax(opt->bounds[2], fmin(opt->bounds[3], new_pos[1]));
        } else {
            for (int j = 0; j < opt->dim; j++) {
                if (r1 < TRADE_OFF) {
                    new_pos[j] = opt->population[i].position[j] + L * (opt->best_solution.position[j] - opt->population[i].position[j]);
                } else {
                    new_pos[j] = opt->population[i].position[j] + r2 * bound_range[j] * rand_double(-0.1, 0.1);
                }
                new_pos[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_pos[j]));
            }
        }

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
}

// Mating Phase (optimized for efficiency)
void SWO_mating_phase(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || !opt->population || !opt->bounds) return;

    double *new_pos = (double *)malloc(opt->dim * sizeof(double));
    if (!new_pos) return;

    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            free(new_pos);
            return;
        }

        int mate_idx = rand() % opt->population_size;
        if (mate_idx == i || mate_idx >= opt->population_size || !opt->population[mate_idx].position) continue;

        // Unroll loop for dim=2
        if (opt->dim == 2) {
            new_pos[0] = (rand_double(0.0, 1.0) < CROSSOVER_PROB) ?
                         opt->population[i].position[0] + rand_double(0.0, 1.0) * (opt->population[mate_idx].position[0] - opt->population[i].position[0]) :
                         opt->population[i].position[0];
            new_pos[1] = (rand_double(0.0, 1.0) < CROSSOVER_PROB) ?
                         opt->population[i].position[1] + rand_double(0.0, 1.0) * (opt->population[mate_idx].position[1] - opt->population[i].position[1]) :
                         opt->population[i].position[1];
            new_pos[0] = fmax(opt->bounds[0], fmin(opt->bounds[1], new_pos[0]));
            new_pos[1] = fmax(opt->bounds[2], fmin(opt->bounds[3], new_pos[1]));
        } else {
            for (int j = 0; j < opt->dim; j++) {
                new_pos[j] = (rand_double(0.0, 1.0) < CROSSOVER_PROB) ?
                             opt->population[i].position[j] + rand_double(0.0, 1.0) * (opt->population[mate_idx].position[j] - opt->population[i].position[j]) :
                             opt->population[i].position[j];
                new_pos[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_pos[j]));
            }
        }

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
}

// Population Reduction (optimized sorting and memory handling)
void SWO_population_reduction(Optimizer *opt, int iter) {
    if (!opt || !opt->population || !opt->population[0].position) return;

    double T = (double)opt->max_iter;
    int current_population = MIN_POPULATION + (int)((opt->population_size - MIN_POPULATION) * ((T - iter) / T));
    if (current_population < MIN_POPULATION) current_population = MIN_POPULATION;

    if (current_population < opt->population_size) {
        // Use insertion sort for small populations (faster for small n)
        for (int i = 1; i < opt->population_size; i++) {
            double key_fitness = opt->population[i].fitness;
            double *key_pos = (double *)malloc(opt->dim * sizeof(double));
            if (!key_pos) return;
            memcpy(key_pos, opt->population[i].position, opt->dim * sizeof(double));

            int j = i - 1;
            while (j >= 0 && opt->population[j].fitness > key_fitness) {
                opt->population[j + 1].fitness = opt->population[j].fitness;
                memcpy(opt->population[j + 1].position, opt->population[j].position, opt->dim * sizeof(double));
                j--;
            }
            opt->population[j + 1].fitness = key_fitness;
            memcpy(opt->population[j + 1].position, key_pos, opt->dim * sizeof(double));
            free(key_pos);
        }

        int old_population_size = opt->population_size;
        opt->population_size = current_population;

        // Reallocate population
        Solution *new_population = (Solution *)realloc(opt->population, opt->population_size * sizeof(Solution));
        if (!new_population) {
            opt->population_size = old_population_size;
            return;
        }
        opt->population = new_population;

        // Allocate and copy positions
        double *new_positions = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
        if (!new_positions) {
            opt->population_size = old_population_size;
            return;
        }

        for (int i = 0; i < opt->population_size; i++) {
            memcpy(new_positions + (i * opt->dim), opt->population[i].position, opt->dim * sizeof(double));
        }

        if (opt->population[0].position) free(opt->population[0].position);
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].position = new_positions + (i * opt->dim);
        }
    }

    enforce_bound_constraints(opt);
}

// Main Optimization Function
void SWO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || !opt->population || !opt->best_solution.position) return;

    srand((unsigned int)time(NULL));

    // Initialize population fitness
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) return;
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
        }
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        if (rand_double(0.0, 1.0) < TRADE_OFF) {
            SWO_hunting_phase(opt, objective_function);
        } else {
            SWO_mating_phase(opt, objective_function);
        }
        SWO_population_reduction(opt, iter);
    }
}
