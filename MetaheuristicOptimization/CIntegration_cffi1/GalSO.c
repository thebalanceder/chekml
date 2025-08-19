#include "GalSO.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// 🔧 Fast Linear Congruential Generator (LCG) for random numbers
static unsigned long lcg_state;
static inline double fast_rand() {
    lcg_state = 6364136223846793005UL * lcg_state + 1442695040888963407UL;
    return (double)(lcg_state >> 32) / 4294967296.0; // Normalize to [0, 1)
}

// 🚀 Main Optimization Function
void GalSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    lcg_state = time(NULL); // Initialize LCG seed

    const int pop_size = DEFAULT_POP_SIZE;
    const int subpop = DEFAULT_SUBPOP;
    const int trials = DEFAULT_TRIALS;
    const int epoch_number = DEFAULT_EPOCH_NUMBER;
    const int iteration1 = DEFAULT_ITERATION1;
    const int iteration2 = DEFAULT_ITERATION2;
    const int dim = opt->dim;
    const int total_particles = subpop * pop_size;

    // Preallocate all memory
    double results[DEFAULT_TRIALS];
    double galaxy_x[2]; // Stack allocation for dim=2
    double galaxy_c = INFINITY;
    long function_calls = 0;

    // Single contiguous array for particles: [x, v, pbest, cost, pbest_c] per particle
    double *particle_data = (double *)malloc(total_particles * (dim * 3 + 2) * sizeof(double));
    double *xgbest = (double *)malloc(subpop * dim * sizeof(double));
    double cgbest[DEFAULT_SUBPOP];
    double l2_particles[DEFAULT_SUBPOP * (2 * 3 + 2)]; // [x, v, pbest, cost, pbest_c] for level 2
    if (!particle_data || !xgbest) exit(1);

    // Statistics accumulators
    double obj_sum = 0.0, obj_sum_sq = 0.0, obj_min = INFINITY, obj_max = -INFINITY;

    // Cache bounds
    const double lb0 = opt->bounds[0], ub0 = opt->bounds[1];
    const double lb1 = opt->bounds[2], ub1 = opt->bounds[3];

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

        // Initialize particles
        for (int s = 0; s < subpop; s++) {
            cgbest[s] = INFINITY;
            for (int p = 0; p < pop_size; p++) {
                int idx = s * pop_size + p;
                int base = idx * (dim * 3 + 2);
                double *x = particle_data + base;
                double *v = x + dim;
                double *pbest = v + dim;
                double *cost = pbest + dim;
                double *pbest_c = cost + 1;

                // Map to population
                x[0] = opt->population[idx].position[0];
                x[1] = opt->population[idx].position[1];
                v[0] = lb0 + fast_rand() * (ub0 - lb0);
                v[1] = lb1 + fast_rand() * (ub1 - lb1);
                pbest[0] = x[0];
                pbest[1] = x[1];
                *cost = opt->population[idx].fitness = objective_function(x);
                *pbest_c = *cost;
                function_calls++;

                // Update subswarm best
                if (*pbest_c < cgbest[s]) {
                    cgbest[s] = *pbest_c;
                    memcpy(xgbest + s * dim, pbest, dim * sizeof(double));
                }
            }
        }

        // Initialize global best
        memcpy(galaxy_x, xgbest, dim * sizeof(double));
        galaxy_c = cgbest[0];
        for (int s = 1; s < subpop; s++) {
            if (cgbest[s] < galaxy_c) {
                galaxy_c = cgbest[s];
                memcpy(galaxy_x, xgbest + s * dim, dim * sizeof(double));
            }
        }
        if (galaxy_c < opt->best_solution.fitness) {
            opt->best_solution.fitness = galaxy_c;
            memcpy(opt->best_solution.position, galaxy_x, dim * sizeof(double));
        }

        // Main optimization loop
        for (int epoch = 0; epoch < epoch_number; epoch++) {
            // Level 1: Subswarm optimization
            const double inv_iter1 = 1.0 / (iteration1 + 1);
            for (int s = 0; s < subpop; s++) {
                for (int r = 0; r <= iteration1; r++) {
                    double c1 = C1_MAX * fast_rand();
                    double c2 = C2_MAX * fast_rand();
                    double inertia = 1.0 - r * inv_iter1;

                    for (int p = 0; p < pop_size; p++) {
                        int idx = s * pop_size + p;
                        int base = idx * (dim * 3 + 2);
                        double *x = particle_data + base;
                        double *v = x + dim;
                        double *pbest = v + dim;
                        double *cost = pbest + dim;
                        double *pbest_c = cost + 1;
                        double *gbest = xgbest + s * dim;

                        // Update velocity and position (unrolled for dim=2)
                        double r1 = -1.0 + 2.0 * fast_rand();
                        double r2 = -1.0 + 2.0 * fast_rand();
                        double v0 = c1 * r1 * (pbest[0] - x[0]) + c2 * r2 * (gbest[0] - x[0]);
                        double v1 = c1 * r1 * (pbest[1] - x[1]) + c2 * r2 * (gbest[1] - x[1]);
                        v[0] = inertia * v[0] + v0;
                        v[1] = inertia * v[1] + v1;
                        v[0] = fmin(fmax(v[0], lb0), ub0);
                        v[1] = fmin(fmax(v[1], lb1), ub1);
                        x[0] += v[0];
                        x[1] += v[1];
                        x[0] = fmin(fmax(x[0], lb0), ub0);
                        x[1] = fmin(fmax(x[1], lb1), ub1);

                        // Update fitness
                        *cost = opt->population[idx].fitness = objective_function(x);
                        function_calls++;

                        // Update personal and subswarm best
                        if (*cost < *pbest_c) {
                            pbest[0] = x[0];
                            pbest[1] = x[1];
                            *pbest_c = *cost;
                            if (*pbest_c < cgbest[s]) {
                                cgbest[s] = *pbest_c;
                                gbest[0] = pbest[0];
                                gbest[1] = pbest[1];
                                if (cgbest[s] < galaxy_c) {
                                    galaxy_c = cgbest[s];
                                    galaxy_x[0] = gbest[0];
                                    galaxy_x[1] = gbest[1];
                                    if (galaxy_c < opt->best_solution.fitness) {
                                        opt->best_solution.fitness = galaxy_c;
                                        opt->best_solution.position[0] = galaxy_x[0];
                                        opt->best_solution.position[1] = galaxy_x[1];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Level 2: Global optimization
            const double inv_iter2 = 1.0 / (iteration2 + 1);
            for (int s = 0; s < subpop; s++) {
                int base = s * (dim * 3 + 2);
                double *x = l2_particles + base;
                double *v = x + dim;
                double *pbest = v + dim;
                double *cost = pbest + dim;
                double *pbest_c = cost + 1;

                x[0] = xgbest[s * dim];
                x[1] = xgbest[s * dim + 1];
                *cost = cgbest[s];
                pbest[0] = x[0];
                pbest[1] = x[1];
                *pbest_c = *cost;
                v[0] = x[0];
                v[1] = x[1];
            }

            for (int r = 0; r <= iteration2; r++) {
                double c3 = C3_MAX * fast_rand();
                double c4 = C4_MAX * fast_rand();
                double inertia = 1.0 - r * inv_iter2;

                for (int s = 0; s < subpop; s++) {
                    int base = s * (dim * 3 + 2);
                    double *x = l2_particles + base;
                    double *v = x + dim;
                    double *pbest = v + dim;
                    double *cost = pbest + dim;
                    double *pbest_c = cost + 1;

                    // Update velocity and position (unrolled for dim=2)
                    double r3 = -1.0 + 2.0 * fast_rand();
                    double r4 = -1.0 + 2.0 * fast_rand();
                    double v0 = c3 * r3 * (pbest[0] - x[0]) + c4 * r4 * (galaxy_x[0] - x[0]);
                    double v1 = c3 * r3 * (pbest[1] - x[1]) + c4 * r4 * (galaxy_x[1] - x[1]);
                    v[0] = inertia * v[0] + v0;
                    v[1] = inertia * v[1] + v1;
                    v[0] = fmin(fmax(v[0], lb0), ub0);
                    v[1] = fmin(fmax(v[1], lb1), ub1);
                    x[0] += v[0];
                    x[1] += v[1];
                    x[0] = fmin(fmax(x[0], lb0), ub0);
                    x[1] = fmin(fmax(x[1], lb1), ub1);

                    // Update fitness
                    *cost = objective_function(x);
                    function_calls++;

                    // Update personal and global best
                    if (*cost < *pbest_c) {
                        pbest[0] = x[0];
                        pbest[1] = x[1];
                        *pbest_c = *cost;
                        if (*pbest_c < galaxy_c) {
                            galaxy_c = *pbest_c;
                            galaxy_x[0] = pbest[0];
                            galaxy_x[1] = pbest[1];
                            if (galaxy_c < opt->best_solution.fitness) {
                                opt->best_solution.fitness = galaxy_c;
                                opt->best_solution.position[0] = galaxy_x[0];
                                opt->best_solution.position[1] = galaxy_x[1];
                            }
                        }
                    }
                }
            }

            printf("Trial=%d Epoch=%d objfun_val=%.6e\n", trial + 1, epoch + 1, galaxy_c);
        }

        results[trial] = galaxy_c;
        obj_sum += galaxy_c;
        obj_sum_sq += galaxy_c * galaxy_c;
        obj_min = fmin(obj_min, galaxy_c);
        obj_max = fmax(obj_max, galaxy_c);
    }

    // 📊 Compute statistics
    double obj_mean = obj_sum / trials;
    double obj_var = (obj_sum_sq / trials) - (obj_mean * obj_mean);
    double obj_std = sqrt(obj_var);
    double obj_median = results[trials / 2]; // Approximate median
    double obj_mode = obj_median; // Simplified

    printf("\nobj_mean=%.6e\n", obj_mean);
    printf("obj_std=%.6e\n", obj_std);
    printf("obj_var=%.6e\n", obj_var);
    printf("best_val=%.6e\n", obj_min);
    printf("worst_val=%.6e\n", obj_max);
    printf("median=%.6e\n", obj_median);
    printf("mode=%.6e\n", obj_mode);
    printf("function_calls=%ld\n", function_calls);

    free(particle_data);
    free(xgbest);
}
