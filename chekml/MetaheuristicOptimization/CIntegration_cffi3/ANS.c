/* ANS.c - GPU-Optimized Artificial Neighborhood Search */
#include "ANS.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// OpenCL kernel for ANS
static const char* ans_kernel_source =
"// Simple LCG random number generator\n"
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
"// Initialize neighborhoods\n"
"__kernel void initialize_neighborhoods(\n"
"    __global float* populations,\n"
"    __global const float* bounds,\n"
"    const int num_vars,\n"
"    const int num_neighborhoods,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < num_neighborhoods * num_vars) {\n"
"        int n = id / num_vars;\n"
"        int d = id % num_vars;\n"
"        uint local_seed = seed + id;\n"
"        float lower = bounds[2 * d];\n"
"        float upper = bounds[2 * d + 1];\n"
"        populations[n * num_vars + d] = lcg_rand_float(&local_seed, lower, upper);\n"
"    }\n"
"}\n"
"// Move neighborhoods\n"
"__kernel void move_neighborhoods(\n"
"    __global float* populations,\n"
"    __global const float* bounds,\n"
"    __global int* neighbor_indices,\n"
"    const int num_vars,\n"
"    const int num_neighborhoods)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < num_neighborhoods * num_vars) {\n"
"        int n = id / num_vars;\n"
"        int d = id % num_vars;\n"
"        int neighbor_idx = neighbor_indices[n];\n"
"        float p = populations[n * num_vars + d];\n"
"        float neighbor = populations[neighbor_idx * num_vars + d];\n"
"        float direction = neighbor - p;\n"
"        float new_pos = p + 0.1f * direction;\n"
"        float lower = bounds[2 * d];\n"
"        float upper = bounds[2 * d + 1];\n"
"        populations[n * num_vars + d] = clamp(new_pos, lower, upper);\n"
"    }\n"
"}\n";

void ANS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, move_kernel = NULL;
    cl_mem bounds_buffer = NULL, neighbor_indices_buffer = NULL;

    // Validate input
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(EXIT_FAILURE);
    }

    const int num_vars = opt->dim;
    const int num_neighborhoods = ANS_NUM_NEIGHBORHOODS;
    const int max_iterations = opt->max_iter;
    if (num_vars < 1 || num_neighborhoods > opt->population_size) {
        fprintf(stderr, "Error: Invalid num_vars (%d) or num_neighborhoods (%d) > population_size (%d)\n", 
                num_vars, num_neighborhoods, opt->population_size);
        exit(EXIT_FAILURE);
    }

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < num_vars; i++) {
        opt->best_solution.position[i] = 0.0;
    }

    // Create OpenCL program
    program = clCreateProgramWithSource(opt->context, 1, &ans_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating ANS program: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building ANS program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_neighborhoods", &err);
    move_kernel = clCreateKernel(program, "move_neighborhoods", &err);
    if (err != CL_SUCCESS || !init_kernel || !move_kernel) {
        fprintf(stderr, "Error creating ANS kernels: %d\n", err);
        clReleaseKernel(init_kernel);
        clReleaseKernel(move_kernel);
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    // Create buffers
    float* bounds_float = (float*)malloc(2 * num_vars * sizeof(float));
    if (!bounds_float) {
        fprintf(stderr, "Error: Memory allocation failed for bounds_float\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * num_vars; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }

    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, 2 * num_vars * sizeof(float), NULL, &err);
    neighbor_indices_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, num_neighborhoods * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !neighbor_indices_buffer) {
        fprintf(stderr, "Error creating ANS buffers: %d\n", err);
        free(bounds_float);
        goto cleanup;
    }

    // Write bounds
    err = clEnqueueWriteBuffer(opt->queue, bounds_buffer, CL_TRUE, 0, 2 * num_vars * sizeof(float), bounds_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bounds buffer: %d\n", err);
        free(bounds_float);
        goto cleanup;
    }

    // Initialize neighborhoods
    uint seed = (uint)time(NULL);
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(int), &num_vars);
    err |= clSetKernelArg(init_kernel, 3, sizeof(int), &num_neighborhoods);
    err |= clSetKernelArg(init_kernel, 4, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting init kernel args: %d\n", err);
        free(bounds_float);
        goto cleanup;
    }

    size_t global_work_size = num_neighborhoods * num_vars;
    err = clEnqueueNDRangeKernel(opt->queue, init_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel: %d\n", err);
        free(bounds_float);
        goto cleanup;
    }

    // Main loop
    float* populations = (float*)malloc(num_neighborhoods * num_vars * sizeof(float));
    float* fitness = (float*)malloc(num_neighborhoods * sizeof(float));
    int* neighbor_indices = (int*)malloc(num_neighborhoods * sizeof(int));
    if (!populations || !fitness || !neighbor_indices) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(populations);
        free(fitness);
        free(neighbor_indices);
        free(bounds_float);
        goto cleanup;
    }

    int best_idx = 0;
    double best_fit = INFINITY;

    for (int iter = 0; iter < max_iterations; iter++) {
        // Read populations
        err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                                  num_neighborhoods * num_vars * sizeof(float), populations, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading populations buffer: %d\n", err);
            goto cleanup_loop;
        }

        // Evaluate fitness
        #pragma omp parallel for
        for (int i = 0; i < num_neighborhoods; i++) {
            double* pos_double = (double*)malloc(num_vars * sizeof(double));
            if (!pos_double) {
                fprintf(stderr, "Error: Memory allocation failed for pos_double\n");
                exit(EXIT_FAILURE);
            }
            for (int d = 0; d < num_vars; d++) {
                pos_double[d] = (double)populations[i * num_vars + d];
            }
            fitness[i] = (float)objective_function(pos_double);
            free(pos_double);
        }

        // Write fitness
        err = clEnqueueWriteBuffer(opt->queue, opt->fitness_buffer, CL_TRUE, 0, 
                                   num_neighborhoods * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            goto cleanup_loop;
        }

        // Find best solution
        best_fit = INFINITY;
        for (int i = 0; i < num_neighborhoods; i++) {
            if (fitness[i] < best_fit) {
                best_fit = fitness[i];
                best_idx = i;
            }
        }

#if VERBOSE
        printf("ANS Iteration %d: Best Fitness = %f\n", iter + 1, best_fit);
#endif

        // Generate random neighbor indices
        for (int i = 0; i < num_neighborhoods; i++) {
            int neighbor_index;
            do {
                neighbor_index = rand() % num_neighborhoods;
            } while (neighbor_index == i);
            neighbor_indices[i] = neighbor_index;
        }

        err = clEnqueueWriteBuffer(opt->queue, neighbor_indices_buffer, CL_TRUE, 0, 
                                   num_neighborhoods * sizeof(int), neighbor_indices, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing neighbor indices buffer: %d\n", err);
            goto cleanup_loop;
        }

        // Move neighborhoods
        err = clSetKernelArg(move_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(move_kernel, 1, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(move_kernel, 2, sizeof(cl_mem), &neighbor_indices_buffer);
        err |= clSetKernelArg(move_kernel, 3, sizeof(int), &num_vars);
        err |= clSetKernelArg(move_kernel, 4, sizeof(int), &num_neighborhoods);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting move kernel args: %d\n", err);
            goto cleanup_loop;
        }

        global_work_size = num_neighborhoods * num_vars;
        err = clEnqueueNDRangeKernel(opt->queue, move_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing move kernel: %d\n", err);
            goto cleanup_loop;
        }
    }

    // Final best solution
    err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                              num_neighborhoods * num_vars * sizeof(float), populations, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final populations buffer: %d\n", err);
        goto cleanup_loop;
    }

    for (int d = 0; d < num_vars; d++) {
        opt->best_solution.position[d] = (double)populations[best_idx * num_vars + d];
    }
    opt->best_solution.fitness = best_fit;

cleanup_loop:
    free(populations);
    free(fitness);
    free(neighbor_indices);

cleanup:
    free(bounds_float);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (neighbor_indices_buffer) clReleaseMemObject(neighbor_indices_buffer);
    clReleaseKernel(init_kernel);
    clReleaseKernel(move_kernel);
    clReleaseProgram(program);
    clFinish(opt->queue);
}
