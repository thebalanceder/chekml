#include "GalSO.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

// 🌌 Structure for a Particle (extends Solution with velocity and personal best)
typedef struct {
    double *x;        // Current position (points to population[i].position)
    double *v;        // Current velocity
    double *pbest;    // Personal best position
    double cost;      // Current cost (population[i].fitness)
    double pbest_c;   // Personal best cost
} Particle;

// 🌌 Structure for a Subswarm
typedef struct {
    Particle *particles; // Array of particles
    double *xgbest;      // Global best position in subswarm
    double cgbest;       // Global best cost in subswarm
} Subswarm;

// 🛠 Initialize a single trial
void initialize_trial(Optimizer *opt, Subswarm *galaxies, double *velocities, double *pbest_positions, double *galaxy_x, double (*objective_function)(double *), long *function_calls) {
    int pop_size = DEFAULT_POP_SIZE;
    int subpop = DEFAULT_SUBPOP;
    int dim = opt->dim;

    #pragma omp parallel for
    for (int i = 0; i < subpop; i++) {
        galaxies[i].particles = (Particle *)malloc(pop_size * sizeof(Particle));
        galaxies[i].xgbest = (double *)malloc(dim * sizeof(double));
        if (!galaxies[i].particles || !galaxies[i].xgbest) exit(1);

        for (int p = 0; p < pop_size; p++) {
            int idx = i * pop_size + p;
            if (idx >= opt->population_size) break;

            Particle *part = &galaxies[i].particles[p];
            part->x = opt->population[idx].position;
            part->v = velocities + (idx * dim);
            part->pbest = pbest_positions + (idx * dim);
            for (int j = 0; j < dim; j++) {
                part->v[j] = opt->bounds[2 * j] + ((double)rand() / RAND_MAX) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
                part->pbest[j] = part->x[j];
            }
            part->cost = opt->population[idx].fitness = objective_function(part->x);
            part->pbest_c = part->cost;
            #pragma omp atomic
            (*function_calls)++;
        }

        // Find subswarm's global best
        memcpy(galaxies[i].xgbest, galaxies[i].particles[0].pbest, dim * sizeof(double));
        galaxies[i].cgbest = galaxies[i].particles[0].pbest_c;
        for (int p = 1; p < pop_size; p++) {
            if (galaxies[i].particles[p].pbest_c < galaxies[i].cgbest) {
                memcpy(galaxies[i].xgbest, galaxies[i].particles[p].pbest, dim * sizeof(double));
                galaxies[i].cgbest = galaxies[i].particles[p].pbest_c;
            }
        }
    }

    // Initialize global best (galaxy_x, galaxy_c)
    memcpy(galaxy_x, galaxies[0].xgbest, dim * sizeof(double));
    double galaxy_c = galaxies[0].cgbest;
    for (int i = 1; i < subpop; i++) {
        if (galaxies[i].cgbest < galaxy_c) {
            memcpy(galaxy_x, galaxies[i].xgbest, dim * sizeof(double));
            galaxy_c = galaxies[i].cgbest;
        }
    }

    // Update Optimizer's best_solution
    if (galaxy_c < opt->best_solution.fitness) {
        opt->best_solution.fitness = galaxy_c;
        memcpy(opt->best_solution.position, galaxy_x, dim * sizeof(double));
    }
}

// 🛠 Free trial memory
void free_trial(Optimizer *opt, Subswarm *galaxies) {
    int subpop = DEFAULT_SUBPOP;
    for (int i = 0; i < subpop; i++) {
        free(galaxies[i].particles);
        free(galaxies[i].xgbest);
    }
}

// 🌌 Level 1 Optimization (Within Subswarms)
void level1_optimization(Optimizer *opt, Subswarm *galaxies, double *galaxy_x, double *galaxy_c, double (*objective_function)(double *), long *function_calls) {
    int pop_size = DEFAULT_POP_SIZE;
    int subpop = DEFAULT_SUBPOP;
    int iteration1 = DEFAULT_ITERATION1;
    int dim = opt->dim;

    #pragma omp parallel for
    for (int num = 0; num < subpop; num++) {
        Subswarm *swarm = &galaxies[num];
        for (int r = 0; r <= iteration1; r++) {
            double c1 = C1_MAX * ((double)rand() / RAND_MAX);
            double c2 = C2_MAX * ((double)rand() / RAND_MAX);
            double inertia = 1.0 - (double)r / (iteration1 + 1);

            for (int p = 0; p < pop_size; p++) {
                int idx = num * pop_size + p;
                if (idx >= opt->population_size) break;

                Particle *part = &swarm->particles[p];
                double r1 = -1.0 + 2.0 * ((double)rand() / RAND_MAX);
                double r2 = -1.0 + 2.0 * ((double)rand() / RAND_MAX);

                for (int j = 0; j < dim; j++) {
                    double lower = opt->bounds[2 * j];
                    double upper = opt->bounds[2 * j + 1];
                    double v = c1 * r1 * (part->pbest[j] - part->x[j]) +
                               c2 * r2 * (swarm->xgbest[j] - part->x[j]);
                    part->v[j] = inertia * part->v[j] + v;
                    if (part->v[j] < lower) part->v[j] = lower;
                    if (part->v[j] > upper) part->v[j] = upper;
                    part->x[j] += part->v[j];
                }

                enforce_bound_constraints(opt);
                part->cost = opt->population[idx].fitness = objective_function(part->x);
                #pragma omp atomic
                (*function_calls)++;

                if (part->cost < part->pbest_c) {
                    memcpy(part->pbest, part->x, dim * sizeof(double));
                    part->pbest_c = part->cost;
                    if (part->pbest_c < swarm->cgbest) {
                        memcpy(swarm->xgbest, part->pbest, dim * sizeof(double));
                        swarm->cgbest = part->pbest_c;
                        #pragma omp critical
                        {
                            if (swarm->cgbest < *galaxy_c) {
                                memcpy(galaxy_x, swarm->xgbest, dim * sizeof(double));
                                *galaxy_c = swarm->cgbest;
                                if (*galaxy_c < opt->best_solution.fitness) {
                                    opt->best_solution.fitness = *galaxy_c;
                                    memcpy(opt->best_solution.position, galaxy_x, dim * sizeof(double));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// 🌌 Level 2 Optimization (Across Subswarms)
void level2_optimization(Optimizer *opt, Subswarm *galaxies, double *galaxy_x, double *galaxy_c, double (*objective_function)(double *), long *function_calls, double *l2_velocities, double *l2_pbest_positions) {
    int subpop = DEFAULT_SUBPOP;
    int iteration2 = DEFAULT_ITERATION2;
    int dim = opt->dim;

    Particle *particles = (Particle *)malloc(subpop * sizeof(Particle));
    if (!particles) exit(1);

    for (int p = 0; p < subpop; p++) {
        particles[p].x = (double *)malloc(dim * sizeof(double));
        if (!particles[p].x) exit(1);
        particles[p].v = l2_velocities + (p * dim);
        particles[p].pbest = l2_pbest_positions + (p * dim);
        memcpy(particles[p].x, galaxies[p].xgbest, dim * sizeof(double));
        particles[p].cost = galaxies[p].cgbest;
        memcpy(particles[p].pbest, galaxies[p].xgbest, dim * sizeof(double));
        particles[p].pbest_c = galaxies[p].cgbest;
        memcpy(particles[p].v, galaxies[p].xgbest, dim * sizeof(double));
    }

    for (int r = 0; r <= iteration2; r++) {
        double c3 = C3_MAX * ((double)rand() / RAND_MAX);
        double c4 = C4_MAX * ((double)rand() / RAND_MAX);
        double inertia = 1.0 - (double)r / (iteration2 + 1);

        for (int p = 0; p < subpop; p++) {
            Particle *part = &particles[p];
            double r3 = -1.0 + 2.0 * ((double)rand() / RAND_MAX);
            double r4 = -1.0 + 2.0 * ((double)rand() / RAND_MAX);

            for (int j = 0; j < dim; j++) {
                double lower = opt->bounds[2 * j];
                double upper = opt->bounds[2 * j + 1];
                double v = c3 * r3 * (part->pbest[j] - part->x[j]) +
                           c4 * r4 * (galaxy_x[j] - part->x[j]);
                part->v[j] = inertia * part->v[j] + v;
                if (part->v[j] < lower) part->v[j] = lower;
                if (part->v[j] > upper) part->v[j] = upper;
                part->x[j] += part->v[j];
                if (part->x[j] < lower) part->x[j] = lower;
                if (part->x[j] > upper) part->x[j] = upper;
            }

            part->cost = objective_function(part->x);
            #pragma omp atomic
            (*function_calls)++;

            if (part->cost < part->pbest_c) {
                memcpy(part->pbest, part->x, dim * sizeof(double));
                part->pbest_c = part->cost;
                if (part->pbest_c < *galaxy_c) {
                    memcpy(galaxy_x, part->pbest, dim * sizeof(double));
                    *galaxy_c = part->pbest_c;
                    if (*galaxy_c < opt->best_solution.fitness) {
                        opt->best_solution.fitness = *galaxy_c;
                        memcpy(opt->best_solution.position, galaxy_x, dim * sizeof(double));
                    }
                }
            }
        }
    }

    for (int p = 0; p < subpop; p++) {
        free(particles[p].x);
    }
    free(particles);
}

// 🚀 Main Optimization Function
void GalSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand(time(NULL));

    int pop_size = DEFAULT_POP_SIZE;
    int subpop = DEFAULT_SUBPOP;
    int trials = DEFAULT_TRIALS;
    int epoch_number = DEFAULT_EPOCH_NUMBER;
    int dim = opt->dim;

    double *results = (double *)malloc(trials * sizeof(double));
    if (!results) exit(1);
    Subswarm *galaxies = (Subswarm *)malloc(subpop * sizeof(Subswarm));
    if (!galaxies) exit(1);
    double *galaxy_x = (double *)malloc(dim * sizeof(double));
    if (!galaxy_x) exit(1);
    double galaxy_c = INFINITY;
    long function_calls = 0;

    // Allocate contiguous memory once for velocities and personal bests
    double *velocities = (double *)malloc(subpop * pop_size * dim * sizeof(double));
    if (!velocities) exit(1);
    double *pbest_positions = (double *)malloc(subpop * pop_size * dim * sizeof(double));
    if (!pbest_positions) exit(1);
    double *l2_velocities = (double *)malloc(subpop * dim * sizeof(double));
    if (!l2_velocities) exit(1);
    double *l2_pbest_positions = (double *)malloc(subpop * dim * sizeof(double));
    if (!l2_pbest_positions) exit(1);

    // Single pass for statistics
    double obj_sum = 0.0, obj_sum_sq = 0.0, obj_min = INFINITY, obj_max = -INFINITY;

    for (int trial = 0; trial < trials; trial++) {
        // Initialize population fitness
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            function_calls++;
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, dim * sizeof(double));
            }
        }

        initialize_trial(opt, galaxies, velocities, pbest_positions, galaxy_x, objective_function, &function_calls);
        galaxy_c = opt->best_solution.fitness;

        for (int epoch = 0; epoch < epoch_number; epoch++) {
            level1_optimization(opt, galaxies, galaxy_x, &galaxy_c, objective_function, &function_calls);
            level2_optimization(opt, galaxies, galaxy_x, &galaxy_c, objective_function, &function_calls, l2_velocities, l2_pbest_positions);
            printf("Trial=%d Epoch=%d objfun_val=%.6e\n", trial + 1, epoch + 1, galaxy_c);
        }

        results[trial] = galaxy_c;
        obj_sum += galaxy_c;
        obj_sum_sq += galaxy_c * galaxy_c;
        if (galaxy_c < obj_min) obj_min = galaxy_c;
        if (galaxy_c > obj_max) obj_max = galaxy_c;

        free_trial(opt, galaxies);
    }

    // 📊 Compute statistics in one pass
    double obj_mean = obj_sum / trials;
    double obj_var = (obj_sum_sq / trials) - (obj_mean * obj_mean);
    double obj_std = sqrt(obj_var);

    // Approximate median (for simplicity, use middle trials)
    double *sorted_results = (double *)malloc(trials * sizeof(double));
    if (!sorted_results) exit(1);
    memcpy(sorted_results, results, trials * sizeof(double));
    for (int i = 0; i < trials - 1; i++) {
        for (int j = i + 1; j < trials; j++) {
            if (sorted_results[i] > sorted_results[j]) {
                double temp = sorted_results[i];
                sorted_results[i] = sorted_results[j];
                sorted_results[j] = temp;
            }
        }
    }
    double obj_median = (trials % 2 == 0) ?
        (sorted_results[trials / 2 - 1] + sorted_results[trials / 2]) / 2.0 :
        sorted_results[trials / 2];
    double obj_mode = obj_median; // Simplified, using median as fallback

    printf("\nobj_mean=%.6e\n", obj_mean);
    printf("obj_std=%.6e\n", obj_std);
    printf("obj_var=%.6e\n", obj_var);
    printf("best_val=%.6e\n", obj_min);
    printf("worst_val=%.6e\n", obj_max);
    printf("median=%.6e\n", obj_median);
    printf("mode=%.6e\n", obj_mode);
    printf("function_calls=%ld\n", function_calls);

    free(galaxies);
    free(galaxy_x);
    free(results);
    free(velocities);
    free(pbest_positions);
    free(l2_velocities);
    free(l2_pbest_positions);
}
