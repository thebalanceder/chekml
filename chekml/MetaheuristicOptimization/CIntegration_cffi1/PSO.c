/* PSO.c - Extreme Speed Particle Swarm Optimization for CPU */
#include "PSO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <emmintrin.h> // SSE/SSE2
#include <smmintrin.h> // SSE4.1/SSE4.2

// Fast thread-local random number generator (LCG)
static inline double fast_rand(unsigned *seed) {
    *seed = *seed * 1103515245 + 12345;
    return ((unsigned)(*seed >> 16) & 0x7FFF) / 32768.0;
}

// Initialize Swarm (SIMD, Parallelized, Dim=2)
void PSO_initialize_swarm(Optimizer *opt, double *velocities, double *pbest_positions, double *pbest_fitnesses) {
    if (!opt || !velocities || !pbest_positions || !pbest_fitnesses || opt->dim != 2) {
        fprintf(stderr, "PSO_initialize_swarm: Invalid arguments or dim != 2\n");
        return;
    }

    double min0 = opt->bounds[0], max0 = opt->bounds[1];
    double min1 = opt->bounds[2], max1 = opt->bounds[3];
    __m128d bounds_min = _mm_set_pd(min1, min0);
    __m128d bounds_max = _mm_set_pd(max1, max0);
    __m128d bounds_range = _mm_sub_pd(bounds_max, bounds_min);

    #pragma omp parallel
    {
        unsigned seed = (unsigned)time(NULL) ^ omp_get_thread_num();
        #pragma omp for schedule(static)
        for (int i = 0; i < opt->population_size; i++) {
            pbest_fitnesses[i] = INFINITY;
            opt->population[i].fitness = INFINITY;

            // SIMD for 2D positions
            __m128d r = _mm_set_pd(fast_rand(&seed), fast_rand(&seed));
            __m128d pos = _mm_add_pd(bounds_min, _mm_mul_pd(r, bounds_range));
            _mm_storeu_pd(&opt->population[i].position[0], pos);
            _mm_storeu_pd(&pbest_positions[i * 2], pos);
            _mm_storeu_pd(&velocities[i * 2], _mm_setzero_pd());
        }
    }
}

// Update Velocity and Position (SIMD, Parallelized, Dim=2)
void PSO_update_velocity_position(Optimizer *opt, double *velocities, double *pbest_positions) {
    if (!opt || !velocities || !pbest_positions || opt->dim != 2) {
        fprintf(stderr, "PSO_update_velocity_position: Invalid arguments or dim != 2\n");
        return;
    }

    static double w = INERTIA_WEIGHT;
    double min0 = opt->bounds[0], max0 = opt->bounds[1];
    double min1 = opt->bounds[2], max1 = opt->bounds[3];
    __m128d bounds_min = _mm_set_pd(min1, min0);
    __m128d bounds_max = _mm_set_pd(max1, max0);
    __m128d vel_scale = _mm_set1_pd(VELOCITY_SCALE);
    __m128d vel_bound = _mm_mul_pd(_mm_sub_pd(bounds_max, bounds_min), vel_scale);
    __m128d vel_min = _mm_sub_pd(_mm_setzero_pd(), vel_bound); // Fixed: Replace _mm_neg_pd
    __m128d vel_max = vel_bound;
    __m128d personal_c = _mm_set1_pd(PERSONAL_LEARNING);
    __m128d global_c = _mm_set1_pd(GLOBAL_LEARNING);
    __m128d inertia = _mm_set1_pd(w);
    __m128d best_pos = _mm_loadu_pd(opt->best_solution.position);

    #pragma omp parallel
    {
        unsigned seed = (unsigned)time(NULL) ^ omp_get_thread_num();
        #pragma omp for schedule(static)
        for (int i = 0; i < opt->population_size; i++) {
            int idx = i * 2;
            __m128d pos = _mm_loadu_pd(&opt->population[i].position[0]);
            __m128d vel = _mm_loadu_pd(&velocities[idx]);
            __m128d pbest = _mm_loadu_pd(&pbest_positions[idx]);

            // Velocity update: vel = w*vel + c1*r1*(pbest-pos) + c2*r2*(gbest-pos)
            __m128d r1 = _mm_set_pd(fast_rand(&seed), fast_rand(&seed));
            __m128d r2 = _mm_set_pd(fast_rand(&seed), fast_rand(&seed));
            __m128d term1 = _mm_mul_pd(inertia, vel);
            __m128d term2 = _mm_mul_pd(personal_c, _mm_mul_pd(r1, _mm_sub_pd(pbest, pos)));
            __m128d term3 = _mm_mul_pd(global_c, _mm_mul_pd(r2, _mm_sub_pd(best_pos, pos)));
            vel = _mm_add_pd(term1, _mm_add_pd(term2, term3));

            // Clamp velocity
            vel = _mm_max_pd(vel, vel_min);
            vel = _mm_min_pd(vel, vel_max);

            // Update position
            pos = _mm_add_pd(pos, vel);

            // Inline bounds enforcement
            pos = _mm_max_pd(pos, bounds_min);
            pos = _mm_min_pd(pos, bounds_max);

            // Store results
            _mm_storeu_pd(&opt->population[i].position[0], pos);
            _mm_storeu_pd(&velocities[idx], vel);
        }
    }
    w *= INERTIA_DAMPING;
}

// Evaluate Particles and Update Bests (Parallelized, Dim=2)
void PSO_evaluate_particles(Optimizer *opt, double (*objective_function)(double *), 
                           double *pbest_positions, double *pbest_fitnesses) {
    if (!opt || !objective_function || !pbest_positions || !pbest_fitnesses || opt->dim != 2) {
        fprintf(stderr, "PSO_evaluate_particles: Invalid arguments or dim != 2\n");
        return;
    }

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < opt->population_size; i++) {
            double fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = fitness;

            if (fitness < pbest_fitnesses[i]) {
                pbest_fitnesses[i] = fitness;
                _mm_storeu_pd(&pbest_positions[i * 2], _mm_loadu_pd(&opt->population[i].position[0]));
            }

            #pragma omp critical
            {
                if (fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = fitness;
                    _mm_storeu_pd(opt->best_solution.position, _mm_loadu_pd(&opt->population[i].position[0]));
                }
            }
        }
    }
}

// Main Optimization Function
void PSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || opt->dim != 2) {
        fprintf(stderr, "PSO_optimize: Invalid arguments or dim != 2\n");
        return;
    }

    // Contiguous memory allocation
    double *velocities = (double *)malloc(opt->population_size * 2 * sizeof(double));
    double *pbest_positions = (double *)malloc(opt->population_size * 2 * sizeof(double));
    double *pbest_fitnesses = (double *)malloc(opt->population_size * sizeof(double));

    if (!velocities || !pbest_positions || !pbest_fitnesses) {
        fprintf(stderr, "PSO_optimize: Memory allocation failed\n");
        free(velocities);
        free(pbest_positions);
        free(pbest_fitnesses);
        return;
    }

    PSO_initialize_swarm(opt, velocities, pbest_positions, pbest_fitnesses);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        PSO_update_velocity_position(opt, velocities, pbest_positions);
        PSO_evaluate_particles(opt, objective_function, pbest_positions, pbest_fitnesses);
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Cleanup
    free(velocities);
    free(pbest_positions);
    free(pbest_fitnesses);
}
