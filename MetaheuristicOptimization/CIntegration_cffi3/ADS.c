/* ADS.c - GPU-Optimized Adaptive Dimension Search */
#include "ADS.h"
#include <string.h>
#include <float.h>
#include <omp.h>

void ADS_optimize(Optimizer *opt, double (*performance_function)(double *)) {
    cl_int err;

    // Validate input
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position || !performance_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(EXIT_FAILURE);
    }

    const int dim = opt->dim;
    if (dim < 1) {
        fprintf(stderr, "Error: Invalid dimension (%d)\n", dim);
        exit(EXIT_FAILURE);
    }
    const int points_per_dim = 17; // Clenshaw-Curtis level 4
    const int total_points = dim * points_per_dim;

    if (total_points > MAX_COLLOCATION_POINTS || total_points > opt->population_size) {
        fprintf(stderr, "Error: Total points (%d) exceed MAX_COLLOCATION_POINTS (%d) or population_size (%d)\n", 
                total_points, MAX_COLLOCATION_POINTS, opt->population_size);
        exit(EXIT_FAILURE);
    }

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = 0.0;
    }

    // Compute and evaluate reference point
    double* ref_point = (double*)malloc(dim * sizeof(double));
    if (!ref_point) {
        fprintf(stderr, "Error: Memory allocation failed for ref_point\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < dim; i++) {
        ref_point[i] = 0.5 * (opt->bounds[2 * i] + opt->bounds[2 * i + 1]);
    }
    double ref_fitness = performance_function(ref_point);
    if (ref_fitness < opt->best_solution.fitness) {
        opt->best_solution.fitness = ref_fitness;
        memcpy(opt->best_solution.position, ref_point, dim * sizeof(double));
    }

    // Generate points on GPU
    generate_points_on_gpu(opt, points_per_dim, total_points);

    // Read generated points
    float* cc_points = (float*)malloc(total_points * sizeof(float));
    if (!cc_points) {
        fprintf(stderr, "Error: Memory allocation failed for cc_points\n");
        free(ref_point);
        exit(EXIT_FAILURE);
    }
    err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                              total_points * sizeof(float), cc_points, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading points buffer: %d\n", err);
        free(cc_points);
        free(ref_point);
        exit(EXIT_FAILURE);
    }

    // Evaluate points in parallel on host
    double* best_point = (double*)malloc(dim * sizeof(double));
    float* fitness_values = (float*)malloc(total_points * sizeof(float));
    if (!best_point || !fitness_values) {
        fprintf(stderr, "Error: Memory allocation failed for best_point or fitness_values\n");
        free(cc_points);
        free(best_point);
        free(fitness_values);
        free(ref_point);
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel for
    for (int i = 0; i < total_points; i++) {
        int dim_idx = i / points_per_dim;
        double* eval_point = (double*)malloc(dim * sizeof(double));
        if (!eval_point) {
            fprintf(stderr, "Error: Memory allocation failed for eval_point\n");
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < dim; j++) {
            eval_point[j] = (j == dim_idx) ? (double)cc_points[i] : ref_point[j];
        }
        fitness_values[i] = (float)performance_function(eval_point);
        free(eval_point);
    }

    // Write fitness values to GPU buffer
    err = clEnqueueWriteBuffer(opt->queue, opt->fitness_buffer, CL_TRUE, 0, 
                               total_points * sizeof(float), fitness_values, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing fitness buffer: %d\n", err);
        free(fitness_values);
        free(best_point);
        free(cc_points);
        free(ref_point);
        exit(EXIT_FAILURE);
    }

    // Find best solution
    for (int i = 0; i < total_points; i++) {
        if (fitness_values[i] < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)fitness_values[i];
            int dim_idx = i / points_per_dim;
            for (int j = 0; j < dim; j++) {
                best_point[j] = (j == dim_idx) ? (double)cc_points[i] : ref_point[j];
                opt->best_solution.position[j] = best_point[j];
            }
        }
    }

    // Cleanup
    free(fitness_values);
    free(best_point);
    free(cc_points);
    free(ref_point);
}
