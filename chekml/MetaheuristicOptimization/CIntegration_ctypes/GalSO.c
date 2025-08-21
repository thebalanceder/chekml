#include "GalSO.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// 🛠 Generate a random double between min and max
double rand_double(double min, double max);

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

    // Initialize particles and subswarms
    for (int i = 0; i < subpop; i++) {
        galaxies[i].particles = (Particle *)malloc(pop_size * sizeof(Particle));
        if (!galaxies[i].particles) exit(1);
        galaxies[i].xgbest = (double *)malloc(dim * sizeof(double));
        if (!galaxies[i].xgbest) exit(1);

        for (int p = 0; p < pop_size; p++) {
            int idx = i * pop_size + p;
            if (idx >= opt->population_size) break;

            galaxies[i].particles[p].x = opt->population[idx].position;
            galaxies[i].particles[p].v = velocities + (idx * dim);
            galaxies[i].particles[p].pbest = pbest_positions + (idx * dim);
            for (int j = 0; j < dim; j++) {
                galaxies[i].particles[p].v[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
                galaxies[i].particles[p].pbest[j] = galaxies[i].particles[p].x[j];
            }
            galaxies[i].particles[p].cost = opt->population[idx].fitness = objective_function(galaxies[i].particles[p].x);
            galaxies[i].particles[p].pbest_c = galaxies[i].particles[p].cost;
            (*function_calls)++;
        }

        // Find subswarm's global best
        galaxies[i].xgbest = (double *)memcpy(galaxies[i].xgbest, galaxies[i].particles[0].pbest, dim * sizeof(double));
        galaxies[i].cgbest = galaxies[i].particles[0].pbest_c;
        for (int p = 1; p < pop_size; p++) {
            if (galaxies[i].particles[p].pbest_c < galaxies[i].cgbest) {
                galaxies[i].xgbest = (double *)memcpy(galaxies[i].xgbest, galaxies[i].particles[p].pbest, dim * sizeof(double));
                galaxies[i].cgbest = galaxies[i].particles[p].pbest_c;
            }
        }
    }

    // Initialize global best (galaxy_x, galaxy_c)
    galaxy_x = (double *)memcpy(galaxy_x, galaxies[0].xgbest, dim * sizeof(double));
    double galaxy_c = galaxies[0].cgbest;
    for (int i = 1; i < subpop; i++) {
        if (galaxies[i].cgbest < galaxy_c) {
            galaxy_x = (double *)memcpy(galaxy_x, galaxies[i].xgbest, dim * sizeof(double));
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
    int pop_size = DEFAULT_POP_SIZE;
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

    for (int num = 0; num < subpop; num++) {
        for (int r = 0; r <= iteration1; r++) {
            double c1 = C1_MAX * rand_double(0.0, 1.0);
            double c2 = C2_MAX * rand_double(0.0, 1.0);
            for (int p = 0; p < pop_size; p++) {
                int idx = num * pop_size + p;
                if (idx >= opt->population_size) break;

                double r1 = -1.0 + 2.0 * rand_double(0.0, 1.0);
                double r2 = -1.0 + 2.0 * rand_double(0.0, 1.0);
                for (int j = 0; j < dim; j++) {
                    double v = c1 * r1 * (galaxies[num].particles[p].pbest[j] - galaxies[num].particles[p].x[j]) +
                               c2 * r2 * (galaxies[num].xgbest[j] - galaxies[num].particles[p].x[j]);
                    galaxies[num].particles[p].v[j] = (1.0 - (double)r / (iteration1 + 1)) * galaxies[num].particles[p].v[j] + v;
                    // Bound velocity
                    if (galaxies[num].particles[p].v[j] < opt->bounds[2 * j]) galaxies[num].particles[p].v[j] = opt->bounds[2 * j];
                    if (galaxies[num].particles[p].v[j] > opt->bounds[2 * j + 1]) galaxies[num].particles[p].v[j] = opt->bounds[2 * j + 1];
                }
                for (int j = 0; j < dim; j++) {
                    galaxies[num].particles[p].x[j] += galaxies[num].particles[p].v[j];
                }
                enforce_bound_constraints(opt);
                galaxies[num].particles[p].cost = opt->population[idx].fitness = objective_function(galaxies[num].particles[p].x);
                (*function_calls)++;
                if (galaxies[num].particles[p].cost < galaxies[num].particles[p].pbest_c) {
                    memcpy(galaxies[num].particles[p].pbest, galaxies[num].particles[p].x, dim * sizeof(double));
                    galaxies[num].particles[p].pbest_c = galaxies[num].particles[p].cost;
                    if (galaxies[num].particles[p].pbest_c < galaxies[num].cgbest) {
                        memcpy(galaxies[num].xgbest, galaxies[num].particles[p].pbest, dim * sizeof(double));
                        galaxies[num].cgbest = galaxies[num].particles[p].pbest_c;
                        if (galaxies[num].cgbest < *galaxy_c) {
                            memcpy(galaxy_x, galaxies[num].xgbest, dim * sizeof(double));
                            *galaxy_c = galaxies[num].cgbest;
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

// 🌌 Level 2 Optimization (Across Subswarms)
void level2_optimization(Optimizer *opt, Subswarm *galaxies, double *galaxy_x, double *galaxy_c, double (*objective_function)(double *), long *function_calls) {
    int subpop = DEFAULT_SUBPOP;
    int iteration2 = DEFAULT_ITERATION2;
    int dim = opt->dim;

    Particle *particles = (Particle *)malloc(subpop * sizeof(Particle));
    if (!particles) exit(1);
    double *velocities = (double *)malloc(subpop * dim * sizeof(double));
    if (!velocities) exit(1);
    double *pbest_positions = (double *)malloc(subpop * dim * sizeof(double));
    if (!pbest_positions) exit(1);

    for (int p = 0; p < subpop; p++) {
        particles[p].x = (double *)malloc(dim * sizeof(double));
        if (!particles[p].x) exit(1);
        particles[p].v = velocities + (p * dim);
        particles[p].pbest = pbest_positions + (p * dim);
        memcpy(particles[p].x, galaxies[p].xgbest, dim * sizeof(double));
        particles[p].cost = galaxies[p].cgbest;
        memcpy(particles[p].pbest, galaxies[p].xgbest, dim * sizeof(double));
        particles[p].pbest_c = galaxies[p].cgbest;
        memcpy(particles[p].v, galaxies[p].xgbest, dim * sizeof(double));
    }

    for (int r = 0; r <= iteration2; r++) {
        double c3 = C3_MAX * rand_double(0.0, 1.0);
        double c4 = C4_MAX * rand_double(0.0, 1.0);
        for (int p = 0; p < subpop; p++) {
            double r3 = -1.0 + 2.0 * rand_double(0.0, 1.0);
            double r4 = -1.0 + 2.0 * rand_double(0.0, 1.0);
            for (int j = 0; j < dim; j++) {
                double v = c3 * r3 * (particles[p].pbest[j] - particles[p].x[j]) +
                           c4 * r4 * (galaxy_x[j] - particles[p].x[j]);
                particles[p].v[j] = (1.0 - (double)r / (iteration2 + 1)) * particles[p].v[j] + v;
                // Bound velocity
                if (particles[p].v[j] < opt->bounds[2 * j]) particles[p].v[j] = opt->bounds[2 * j];
                if (particles[p].v[j] > opt->bounds[2 * j + 1]) particles[p].v[j] = opt->bounds[2 * j + 1];
            }
            for (int j = 0; j < dim; j++) {
                particles[p].x[j] += particles[p].v[j];
            }
            // Bound position
            for (int j = 0; j < dim; j++) {
                if (particles[p].x[j] < opt->bounds[2 * j]) particles[p].x[j] = opt->bounds[2 * j];
                if (particles[p].x[j] > opt->bounds[2 * j + 1]) particles[p].x[j] = opt->bounds[2 * j + 1];
            }
            particles[p].cost = objective_function(particles[p].x);
            (*function_calls)++;
            if (particles[p].cost < particles[p].pbest_c) {
                memcpy(particles[p].pbest, particles[p].x, dim * sizeof(double));
                particles[p].pbest_c = particles[p].cost;
                if (particles[p].pbest_c < *galaxy_c) {
                    memcpy(galaxy_x, particles[p].pbest, dim * sizeof(double));
                    *galaxy_c = particles[p].pbest_c;
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
    free(velocities);
    free(pbest_positions);
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

    // Allocate contiguous memory for velocities and personal bests
    double *velocities = (double *)malloc(subpop * pop_size * dim * sizeof(double));
    if (!velocities) exit(1);
    double *pbest_positions = (double *)malloc(subpop * pop_size * dim * sizeof(double));
    if (!pbest_positions) exit(1);

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
            level2_optimization(opt, galaxies, galaxy_x, &galaxy_c, objective_function, &function_calls);
            printf("Trial=%d Epoch=%d objfun_val=%.6e\n", trial + 1, epoch + 1, galaxy_c);
        }

        results[trial] = galaxy_c;
        free_trial(opt, galaxies);
    }

    // 📊 Compute statistics
    double obj_mean = 0.0, obj_var = 0.0, obj_min = results[0], obj_max = results[0];
    for (int i = 0; i < trials; i++) {
        obj_mean += results[i];
        if (results[i] < obj_min) obj_min = results[i];
        if (results[i] > obj_max) obj_max = results[i];
    }
    obj_mean /= trials;
    for (int i = 0; i < trials; i++) {
        obj_var += (results[i] - obj_mean) * (results[i] - obj_mean);
    }
    obj_var /= trials;
    double obj_std = sqrt(obj_var);
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
}
