#include "CucS.h"
#include <time.h>

// Initialize RNG seed
static void init_rng(LCG_Rand *restrict rng) {
    static int initialized = 0;
    if (!initialized) {
        rng->state = (uint64_t)time(NULL);
        initialized = 1;
    }
}

// Initialize nests
void initialize_nests(Optimizer *restrict opt) {
    LCG_Rand rng;
    init_rng(&rng);
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double min = opt->bounds[2 * j];
            double max = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = cucs_rand_double(&rng, min, max);
        }
        opt->population[i].fitness = INFINITY;
    }
    fast_enforce_bounds(opt);
}

// Evaluate fitness
void evaluate_nests(Optimizer *restrict opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness;
        if (fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
}

// Generate new solutions via Levy flights
void get_cuckoos(Optimizer *restrict opt) {
    LCG_Rand rng;
    init_rng(&rng);
    for (int i = 0; i < opt->population_size; i++) {
        double *restrict pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double s = pos[j];
            double u = cucs_rand_double(&rng, -1.0, 1.0) * CS_SIGMA;
            double v = cucs_rand_double(&rng, -1.0, 1.0);
            double step = v != 0.0 ? u / pow(fabs(v), 1.0 / CS_BETA) : u;
            double stepsize = CS_STEP_SCALE * step * (s - opt->best_solution.position[j]);
            pos[j] += stepsize * cucs_rand_double(&rng, -1.0, 1.0);
        }
    }
    fast_enforce_bounds(opt);
}

// Replace some nests
void empty_nests(Optimizer *restrict opt) {
    LCG_Rand rng;
    init_rng(&rng);
    int n = opt->population_size;
    int indices[n];

    // Initialize indices
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    // Shuffle indices
    for (int i = n - 1; i > 0; i--) {
        int j = lcg_next(&rng) % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    // Update nests
    for (int i = 0; i < n; i++) {
        int idx = indices[i];
        int idx2 = indices[(i + 1) % n];
        double *restrict pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double r = cucs_rand_double(&rng, 0.0, 1.0);
            double stepsize = (r > CS_PA) ? cucs_rand_double(&rng, 0.0, 1.0) * 
                            (opt->population[idx].position[j] - opt->population[idx2].position[j]) : 0.0;
            pos[j] += stepsize;
        }
    }
    fast_enforce_bounds(opt);
}

// Main optimization function
void CucS_optimize(Optimizer *restrict opt, double (*objective_function)(double *)) {
    opt->population_size = CS_POPULATION_SIZE;
    opt->max_iter = CS_MAX_ITER;
    initialize_nests(opt);
    evaluate_nests(opt, objective_function);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        get_cuckoos(opt);
        evaluate_nests(opt, objective_function);
        empty_nests(opt);
        evaluate_nests(opt, objective_function);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
    printf("Total evaluations=%d\n", opt->max_iter * opt->population_size * 2);
}
