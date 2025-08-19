#include "GPC.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>  // For AVX2 intrinsics
#include "generaloptimizer.h"

// 🎲 Generate a random number in the given range
double random_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// 📌 Pre-allocate memory for movement arrays (d_move, x_move) outside the main loop
void preallocate_memory_for_movement(double*** d_move, double*** x_move, int population_size, int dim) {
    // Allocate memory once for all individuals and dimensions
    *d_move = (double**)malloc(population_size * sizeof(double*));
    *x_move = (double**)malloc(population_size * sizeof(double*));
    
    for (int i = 0; i < population_size; i++) {
        // Align memory to 32-byte boundaries for SIMD operations
        posix_memalign((void**)&((*d_move)[i]), 32, sizeof(double) * 4);  // 4 elements per individual
        posix_memalign((void**)&((*x_move)[i]), 32, sizeof(double) * 4);
    }
}

// 📌 Free allocated memory after optimization process
void free_memory_for_movement(double** d_move, double** x_move, int population_size) {
    for (int i = 0; i < population_size; i++) {
        free(d_move[i]);
        free(x_move[i]);
    }
    free(d_move);
    free(x_move);
}

// 📌 Compute stone and worker movement distances using AVX2 for SIMD optimization
void compute_movement(double velocity, double friction, double* d, double* x) {
    __m256d vel = _mm256_set1_pd(velocity);    // Load the velocity into an AVX2 register (4 elements)
    __m256d fric = _mm256_set1_pd(friction);   // Load the friction into an AVX2 register (4 elements)
    __m256d g = _mm256_set1_pd(G);             // Load the gravitational constant
    __m256d theta = _mm256_set1_pd(THETA);     // Load the angle theta
    
    // Stone movement calculation
    __m256d sin_theta = _mm256_set1_pd(sin(THETA));
    __m256d cos_theta = _mm256_set1_pd(cos(THETA));
    __m256d term1 = _mm256_add_pd(sin_theta, _mm256_mul_pd(fric, cos_theta));
    __m256d stone_d = _mm256_div_pd(_mm256_mul_pd(vel, vel), _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(g, term1)));
    
    // Worker movement calculation
    __m256d worker_x = _mm256_div_pd(_mm256_mul_pd(vel, vel), _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(g, sin_theta)));
    
    // Store the results (for simplicity, just store the first element in the array)
    _mm256_storeu_pd(d, stone_d);
    _mm256_storeu_pd(x, worker_x);
}

// 📊 Evaluate Fitness of Population with Enhanced OpenMP
void evaluate_population(Optimizer* opt, ObjectiveFunction objective_function) {
    // Using dynamic scheduling for better load balancing
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// 📌 Update Population Based on Physics Movement with Reused Memory
void update_population(Optimizer* opt, double** d_move, double** x_move) {
    int best_index = 0;
    double best_fitness = opt->population[0].fitness;
    
    // Find best solution using OpenMP reduction
    #pragma omp parallel for reduction(min:best_fitness) schedule(static)
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < best_fitness) {
            best_fitness = opt->population[i].fitness;
            best_index = i;
        }
    }
    
    // Copy the best solution
    memcpy(opt->best_solution.position, opt->population[best_index].position, opt->dim * sizeof(double));
    opt->best_solution.fitness = best_fitness;
    
    // Parallelize the update of the population with task parallelism
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Parallelizing the population update with tasks
            for (int i = 0; i < opt->population_size; i++) {
                if (i != best_index) {
                    #pragma omp task
                    {
                        for (int d = 0; d < opt->dim; d++) {
                            double velocity = random_uniform(V_MIN, V_MAX);
                            double friction = random_uniform(MU_MIN, MU_MAX);

                            // Use the pre-allocated d_move and x_move arrays
                            double* d_move_ptr = d_move[i];
                            double* x_move_ptr = x_move[i];

                            // **Prefetch data** to improve cache locality
                            _mm_prefetch((char*)d_move_ptr, _MM_HINT_T0);
                            _mm_prefetch((char*)x_move_ptr, _MM_HINT_T0);

                            // Compute the movement
                            compute_movement(velocity, friction, d_move_ptr, x_move_ptr);
                            
                            double epsilon = random_uniform(-0.5 * (V_MAX - V_MIN), 0.5 * (V_MAX - V_MIN));
                            double new_position = (opt->population[i].position[d] + d_move_ptr[0]) * (x_move_ptr[0] * epsilon); // Simplified

                            // Enforce boundary constraints
                            if (new_position < opt->bounds[2 * d]) {
                                new_position = opt->bounds[2 * d];
                            }
                            if (new_position > opt->bounds[2 * d + 1]) {
                                new_position = opt->bounds[2 * d + 1];
                            }
                            
                            opt->population[i].position[d] = new_position;
                        }
                    }
                }
            }
        }
    }
}

// 🚀 Main Optimization Loop with Reused Memory for Movement Arrays
void GPC_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    if (!opt) return;

    // Pre-allocate memory for movement arrays before optimization
    double** d_move = NULL;
    double** x_move = NULL;
    preallocate_memory_for_movement(&d_move, &x_move, opt->population_size, opt->dim);

    // Parallel population initialization with OpenMP
    #pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < opt->population_size; i++) {
        for (int d = 0; d < opt->dim; d++) {
            double min_bound = opt->bounds[2 * d];
            double max_bound = opt->bounds[2 * d + 1];
            opt->population[i].position[d] = random_uniform(min_bound, max_bound);
        }
        opt->population[i].fitness = INFINITY;
    }

    // Optimization loop with reused memory
    for (int iter = 0; iter < opt->max_iter; iter++) {
        evaluate_population(opt, objective_function);
        update_population(opt, d_move, x_move);
    }

    // Free memory after optimization is complete
    free_memory_for_movement(d_move, x_move, opt->population_size);
}