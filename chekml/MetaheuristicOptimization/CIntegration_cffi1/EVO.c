#include "EVO.h"
#include "generaloptimizer.h"
#include <immintrin.h>
#include <string.h>
#include <time.h>

// Fast linear congruential generator
static unsigned long lcg_state = 1;
static inline double fast_rand_double(double min, double max) {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return min + (max - min) * ((double)(lcg_state >> 32) / 0xFFFFFFFFU);
}

// Comparison function for qsort
static int compare_fitness(const void *a, const void *b) {
    double fa = ((double*)a)[0];
    double fb = ((double*)b)[0];
    return (fa > fb) - (fa < fb);
}

// Initialize particle positions and velocities
void evo_initialize_particles(Optimizer *opt, EVO_Particle *particles) {
    const int dim = opt->dim;
    for (int i = 0; i < opt->population_size; i++) {
        double *vel = particles[i].data;
        double *grad = vel + dim;
        double *pos = grad + dim;
        for (int j = 0; j < dim; j++) {
            pos[j] = fast_rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
            vel[j] = fast_rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
            grad[j] = 0.0;
        }
        opt->population[i].position = pos;
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Evaluate fitness for all particles
void evaluate_fitness_evo(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Compute gradient using finite differences
void compute_gradient(Optimizer *opt, EVO_Particle *particles, double (*objective_function)(double *)) {
    double x_plus[MAX_DIM_EVO], x_minus[MAX_DIM_EVO];
    const int dim = opt->dim;

    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        double *grad = particles[i].data + dim;
        memcpy(x_plus, pos, dim * sizeof(double));
        memcpy(x_minus, pos, dim * sizeof(double));
        
        for (int j = 0; j < dim; j++) {
            x_plus[j] += LEARNING_RATE;
            x_minus[j] -= LEARNING_RATE;
            grad[j] = (objective_function(x_plus) - objective_function(x_minus)) / (2 * LEARNING_RATE);
            x_plus[j] = pos[j];
            x_minus[j] = pos[j];
        }
    }
}

// Update velocities and positions with SIMD
void update_velocity_and_position(Optimizer *opt, EVO_Particle *particles) {
    const int dim = opt->dim;
    const __m256d momentum_vec = _mm256_set1_pd(MOMENTUM);
    const __m256d step_size_vec = _mm256_set1_pd(STEP_SIZE);
    const __m256d neg_one_vec = _mm256_set1_pd(-1.0);

    for (int i = 0; i < opt->population_size; i++) {
        double *vel = particles[i].data;
        double *grad = vel + dim;
        double *pos = grad + dim;
        
        // SIMD processing for multiples of 4
        int j = 0;
        for (; j <= dim - 4; j += 4) {
            __m256d vel_vec = _mm256_loadu_pd(vel + j);
            __m256d grad_vec = _mm256_loadu_pd(grad + j);
            __m256d pos_vec = _mm256_loadu_pd(pos + j);
            
            // velocity = momentum * velocity + step_size * gradient
            vel_vec = _mm256_add_pd(_mm256_mul_pd(momentum_vec, vel_vec),
                                   _mm256_mul_pd(step_size_vec, grad_vec));
            // position -= velocity
            pos_vec = _mm256_add_pd(pos_vec, _mm256_mul_pd(neg_one_vec, vel_vec));
            
            _mm256_storeu_pd(vel + j, vel_vec);
            _mm256_storeu_pd(pos + j, pos_vec);
        }
        
        // Scalar processing for remaining elements
        for (; j < dim; j++) {
            vel[j] = MOMENTUM * vel[j] + STEP_SIZE * grad[j];
            pos[j] -= vel[j];
        }
    }
    enforce_bound_constraints(opt);
}

// Free EVO particle arrays
void free_evo_particles(EVO_Particle *particles, int population_size) {
    for (int i = 0; i < population_size; i++) {
        free(particles[i].data);
    }
    free(particles);
}

// Main Optimization Function
void EVO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    lcg_state = (unsigned long)time(NULL); // Seed LCG

    // Allocate EVO-specific particle data
    const int dim = opt->dim;
    EVO_Particle *particles = (EVO_Particle *)malloc(opt->population_size * sizeof(EVO_Particle));
    if (!particles) {
        fprintf(stderr, "EVO_optimize: Memory allocation failed for particles\n");
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        particles[i].data = (double *)malloc(3 * dim * sizeof(double)); // velocity, gradient, position
        if (!particles[i].data) {
            fprintf(stderr, "EVO_optimize: Memory allocation failed for particle %d\n", i);
            free_evo_particles(particles, i);
            return;
        }
    }

    // Initialize reusable arrays
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    int *sorted_indices = (int *)malloc(opt->population_size * sizeof(int));
    double *sort_data = (double *)malloc(opt->population_size * 2 * sizeof(double)); // [fitness, index]
    EVO_Particle *temp_particles = (EVO_Particle *)malloc(opt->population_size * sizeof(EVO_Particle));
    double *temp_fitness = (double *)malloc(opt->population_size * sizeof(double));
    if (!fitness || !sorted_indices || !sort_data || !temp_particles || !temp_fitness) {
        fprintf(stderr, "EVO_optimize: Memory allocation failed for arrays\n");
        free_evo_particles(particles, opt->population_size);
        free(fitness);
        free(sorted_indices);
        free(sort_data);
        free(temp_particles);
        free(temp_fitness);
        return;
    }

    // Initialize particles
    evo_initialize_particles(opt, particles);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness
        evaluate_fitness_evo(opt, objective_function);

        // Update best solution (unrolled for small populations)
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, dim * sizeof(double));
            }
        }

        // Prepare data for sorting
        for (int i = 0; i < opt->population_size; i++) {
            sort_data[2 * i] = opt->population[i].fitness;
            sort_data[2 * i + 1] = i;
        }

        // Sort using qsort
        qsort(sort_data, opt->population_size, 2 * sizeof(double), compare_fitness);

        // Reorder particles and fitness
        for (int i = 0; i < opt->population_size; i++) {
            int idx = (int)sort_data[2 * i + 1];
            temp_particles[i] = particles[idx];
            temp_fitness[i] = sort_data[2 * i];
        }
        for (int i = 0; i < opt->population_size; i++) {
            particles[i] = temp_particles[i];
            opt->population[i].fitness = temp_fitness[i];
            opt->population[i].position = particles[i].data + 2 * dim;
        }

        // Compute gradient
        compute_gradient(opt, particles, objective_function);

        // Update velocities and positions
        update_velocity_and_position(opt, particles);

        // Print progress
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Clean up
    free(fitness);
    free(sorted_indices);
    free(sort_data);
    free(temp_particles);
    free(temp_fitness);
    free_evo_particles(particles, opt->population_size);
}
