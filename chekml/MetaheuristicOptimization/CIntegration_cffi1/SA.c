#include "SA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>

// Fast random double in range [min, max]
static inline double random_uniform(double min, double max) {
    return min + ((double)rand() / (double)RAND_MAX) * (max - min);
}

// Branchless perturbation with clamping
static inline void perturb_solution(double* out, const double* in, int dim, const double* bounds) {
    memcpy(out, in, sizeof(double) * dim);
    int idx = rand() % dim;
    double perturb = ((double)rand() / RAND_MAX - 0.5) / 50.0;
    out[idx] += perturb;

    // Branchless clamp
    double min = bounds[2 * idx];
    double max = bounds[2 * idx + 1];
    out[idx] = fmax(min, fmin(max, out[idx]));
}

void SA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    if (!opt || !objective_function) return;

    int dim = opt->dim;
    int max_iter = opt->max_iter;
    double* bounds = opt->bounds;

    // Allocate once
    double* parent = (double*)malloc(sizeof(double) * dim);
    double* candidate = (double*)malloc(sizeof(double) * dim);
    if (!parent || !candidate) return;

    // Initialize random solution
    for (int d = 0; d < dim; ++d)
        parent[d] = random_uniform(bounds[2 * d], bounds[2 * d + 1]);

    double T = INITIAL_TEMP;
    double old_energy = objective_function(parent);
    memcpy(opt->best_solution.position, parent, sizeof(double) * dim);
    opt->best_solution.fitness = old_energy;

    int itry = 0, success = 0, consec_rej = 0;

    while (1) {
        ++itry;

        // Cooling or stopping criteria
        if (itry >= MAX_TRIES || success >= MAX_SUCCESS) {
            if (T < STOP_TEMP || consec_rej >= MAX_CONSEC_REJ) break;
            T *= COOLING_FACTOR;
            itry = 0;
            success = 0;
        }

        // Create candidate & evaluate
        perturb_solution(candidate, parent, dim, bounds);
        double new_energy = objective_function(candidate);

        // Fail-safe exit for bad energy
        if (new_energy < -INFINITY) {
            memcpy(parent, candidate, sizeof(double) * dim);
            old_energy = new_energy;
            break;
        }

        double delta = old_energy - new_energy;
        int accepted = 0;

        // Accept based on improvement or probabilistic jump
        if (delta > MIN_DELTA || ((double)rand() / RAND_MAX) < exp(delta / (BOLTZMANN_CONST * T))) {
            memcpy(parent, candidate, sizeof(double) * dim);
            old_energy = new_energy;
            accepted = 1;
        }

        if (accepted) {
            ++success;
            consec_rej = 0;
        } else {
            ++consec_rej;
        }

        // Save best if improved
        if (new_energy < opt->best_solution.fitness) {
            memcpy(opt->best_solution.position, candidate, sizeof(double) * dim);
            opt->best_solution.fitness = new_energy;
        }

        enforce_bound_constraints(opt); // optional
    }

    free(parent);
    free(candidate);
}
