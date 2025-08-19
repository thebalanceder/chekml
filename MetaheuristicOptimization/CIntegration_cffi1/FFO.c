#include "FFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed random generator
#include <math.h>
#include <string.h>  // For memcpy

/* Fast inverse square root (Quake III style) for distance calculations */
static inline double fast_inv_sqrt(double x) {
    if (x <= 0.0) return 1e-10;  // Handle zero distance
    double xhalf = 0.5 * x;
    long i = *(long*)&x;           // Bit hack
    i = 0x5f3759df - (i >> 1);    // Magic number
    x = *(double*)&i;
    x = x * (1.5 - xhalf * x * x); // Newton iteration
    return x;                      // Returns 1/sqrt(x)
}

/* Function to generate a random double between min and max */
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

/* Enforce bounds constraints (inlined for speed) */
static inline void enforce_bounds(double *position, const double *bounds, int dim) {
    for (int j = 0; j < dim; j++) {
        if (position[j] < bounds[2 * j]) position[j] = bounds[2 * j];
        else if (position[j] > bounds[2 * j + 1]) position[j] = bounds[2 * j + 1];
    }
}

/* Initialize the fruit fly swarm and its axis */
void initialize_swarm(Optimizer *opt) {
    // 🌟 Stack-based swarm axis (2D: X_axis, Y_axis)
    double swarm_axis[2] = {
        rand_double(opt->bounds[2 * 0], opt->bounds[2 * 0 + 1]),
        rand_double(opt->bounds[2 * 0], opt->bounds[2 * 0 + 1])
    };

    // 🐝 Initialize swarm positions (unrolled for small dims)
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        if (opt->dim == 2) {  // Common case: 2D
            pos[0] = swarm_axis[0] + SMELL_RANGE * (2.0 * rand_double(0.0, 1.0) - 1.0);
            pos[1] = swarm_axis[1] + SMELL_RANGE * (2.0 * rand_double(0.0, 1.0) - 1.0);
        } else {
            for (int j = 0; j < opt->dim; j++) {
                pos[j] = swarm_axis[j % 2] + SMELL_RANGE * (2.0 * rand_double(0.0, 1.0) - 1.0);
            }
        }
        opt->population[i].fitness = INFINITY;
        enforce_bounds(pos, opt->bounds, opt->dim);
    }
}

/* Sm over fruit flies. This loop is not parallelized to avoid threading overhead,
   // as the focus is on single-threaded CPU performance. For multi-core CPUs,
   // OpenMP could be reintroduced if population_size is very large.
   #pragma omp parallel for
   for (int i = 0; i < opt->population_size; i++) {
       double *pos = opt->population[i].position;
       if (opt->dim == 2) {
           pos[0] = swarm_axis[0] + SMELL_RANGE * (2.0 * rand_double(0.0, 1.0) - 1.0);
           pos[1] = swarm_axis[1] + SMELL_RANGE * (2.0 * rand_double(0.0, 1.0) - 1.0);
       } else {
           for (int j = 0; j < opt->dim; j++) {
               pos[j] = swarm_axis[j % 2] + SMELL_RANGE * (2.0 * rand_double(0.0, 1.0) - 1.0);
           }
       }
       enforce_bounds(pos, opt->bounds, opt->dim);
   }
}

/* Evaluate swarm's smell concentration (fitness) */
void evaluate_swarm(Optimizer *opt, double (*objective_function)(double *)) {
    // 🍎 Stack-based buffer for scaled position
    double scaled_position[4];  // Support up to 4D; adjust if dim > 4
    double *scaled_pos = (opt->dim <= 4) ? scaled_position : (double *)malloc(opt->dim * sizeof(double));
    if (opt->dim > 4 && !scaled_pos) {
        fprintf(stderr, "Memory allocation failed for scaled_position\n");
        exit(1);
    }

    // 🐝 Evaluate fitness for each fruit fly
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        double dist_sq = 0.0;
        if (opt->dim == 2) {  // Unroll for 2D
            dist_sq = pos[0] * pos[0] + pos[1] * pos[1];
        } else {
            for (int j = 0; j < opt->dim; j++) {
                dist_sq += pos[j] * pos[j];
            }
        }
        double s = fast_inv_sqrt(dist_sq);  // Fast 1/sqrt(x)

		opt->population[i].fitness = objective_function(opt->population[i].position);

    }

    if (opt->dim > 4) free(scaled_pos);
}

// 🌀 Smell-based local random search
void smell_based_search(Optimizer *opt) {
    int dim = opt->dim;
    double x_axis = opt->best_solution.position[0];
    double y_axis = (dim > 1) ? opt->best_solution.position[1] : x_axis;

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < dim; j++) {
            double offset = SMELL_RANGE * (2.0 * rand_double(0.0, 1.0) - 1.0);
            double base = (j % 2 == 0) ? x_axis : y_axis;
            opt->population[i].position[j] = base + offset;
        }
    }

    enforce_bound_constraints(opt);
}

/* Vision-based update (move swarm to best solution) */
void vision_based_update(Optimizer *opt) {
    // 👀 Find best solution
    double best_fitness = INFINITY;
    int best_idx = 0;
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < best_fitness) {
            best_fitness = opt->population[i].fitness;
            best_idx = i;
        }
    }

    // Update best solution if improved
    if (best_fitness < opt->best_solution.fitness) {
        opt->best_solution.fitness = best_fitness;
        memcpy(opt->best_solution.position, opt->population[best_idx].position,
               opt->dim * sizeof(double));
    }
}

/* Main Optimization Function */
void FFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // 🌱 Initialize random seed
    srand((unsigned int)time(NULL));

    // 🚀 Initialize swarm
    initialize_swarm(opt);
    evaluate_swarm(opt, objective_function);
    vision_based_update(opt);

    // 🔄 Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        smell_based_search(opt);
        evaluate_swarm(opt, objective_function);
        vision_based_update(opt);

        // 📈 Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
