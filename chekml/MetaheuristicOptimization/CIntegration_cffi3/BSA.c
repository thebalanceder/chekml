/* BSA.c - GPU-Optimized Backtracking Search Algorithm */
#include "BSA.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// OpenCL kernel for BSA
static const char* bsa_kernel_source =
"// Simple LCG random number generator\n"
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
"// Initialize population\n"
"__kernel void initialize_population(\n"
"    __global float* population,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size * dim) {\n"
"        int indiv_idx = id / dim;\n"
"        int d = id % dim;\n"
"        uint local_seed = seed + id;\n"
"        float min = bounds[2 * d];\n"
"        float max = bounds[2 * d + 1];\n"
"        population[indiv_idx * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"    }\n"
"}\n"
"// Selection and pseudo-shuffling\n"
"__kernel void selection_shuffle(\n"
"    __global float* population,\n"
"    __global float* historical_pop,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id;\n"
"        // Conditional update of historical population\n"
"        float r1 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float r2 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        if (r1 < r2) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                historical_pop[id * dim + d] = population[id * dim + d];\n"
"            }\n"
"        }\n"
"        // Pseudo-shuffle via random swap\n"
"        int swap_idx = lcg_rand(&local_seed) % population_size;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float temp = historical_pop[id * dim + d];\n"
"            historical_pop[id * dim + d] = historical_pop[swap_idx * dim + d];\n"
"            historical_pop[swap_idx * dim + d] = temp;\n"
"        }\n"
"    }\n"
"}\n"
"// Mutation and boundary control\n"
"#define BSA_DIM_RATE 0.5f\n"
"__kernel void mutation(\n"
"    __global float* population,\n"
"    __global const float* historical_pop,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id;\n"
"        // Cache bounds in local memory\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        // Scale factor (Brownian-walk)\n"
"        float F = 3.0f * lcg_rand_float(&local_seed, -1.0f, 1.0f);\n"
"        // Random number of dimensions to mutate\n"
"        float r1 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float r2 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        int count = (r1 < r2) ? (int)(ceil(BSA_DIM_RATE * lcg_rand_float(&local_seed, 0.0f, 1.0f) * dim)) : 1;\n"
"        float temp[32]; // Assumes dim <= 32\n"
"        for (int d = 0; d < dim; d++) {\n"
"            temp[d] = population[id * dim + d];\n"
"        }\n"
"        // Mutate random dimensions\n"
"        for (int c = 0; c < count; c++) {\n"
"            int j = lcg_rand(&local_seed) % dim;\n"
"            temp[j] += F * (historical_pop[id * dim + j] - population[id * dim + j]);\n"
"        }\n"
"        // Boundary control\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float lower = local_bounds[2 * d];\n"
"            float upper = local_bounds[2 * d + 1];\n"
"            if (temp[d] < lower || temp[d] > upper) {\n"
"                temp[d] = lcg_rand_float(&local_seed, lower, upper);\n"
"            }\n"
"            population[id * dim + d] = temp[d];\n"
"        }\n"
"    }\n"
"}\n"
"// Find best solution index\n"
"__kernel void find_best(\n"
"    __global const float* fitness,\n"
"    __global int* best_idx,\n"
"    const int population_size,\n"
"    __local float* local_fitness,\n"
"    __local int* local_indices)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    if (id < population_size) {\n"
"        local_fitness[local_id] = fitness[id];\n"
"        local_indices[local_id] = id;\n"
"    } else {\n"
"        local_fitness[local_id] = INFINITY;\n"
"        local_indices[local_id] = -1;\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    // Parallel reduction\n"
"    for (int offset = local_size / 2; offset > 0; offset /= 2) {\n"
"        if (local_id < offset) {\n"
"            if (local_fitness[local_id + offset] < local_fitness[local_id]) {\n"
"                local_fitness[local_id] = local_fitness[local_id + offset];\n"
"                local_indices[local_id] = local_indices[local_id + offset];\n"
"            }\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (local_id == 0) {\n"
"        best_idx[get_group_id(0)] = local_indices[0];\n"
"    }\n"
"}\n";

void BSA_optimize(Optimizer* opt, double (*objective_function)(double*)) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, selection_kernel = NULL, mutation_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, historical_pop_buffer = NULL, fitness_buffer = NULL, best_idx_buffer = NULL;
    float* bounds_float = NULL, *population = NULL, *fitness = NULL;
    int* best_idx_array = NULL;

    // Validate input
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(EXIT_FAILURE);
    }

    const int dim = opt->dim;
    const int population_size = opt->population_size;
    const int max_iter = opt->max_iter;
    if (dim < 1 || population_size < 1 || max_iter < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), or max_iter (%d)\n", 
                dim, population_size, max_iter);
        exit(EXIT_FAILURE);
    }
    if (dim > 32) {
        fprintf(stderr, "Error: dim (%d) exceeds maximum supported value of 32\n", dim);
        exit(EXIT_FAILURE);
    }

    // Query device capabilities
    size_t max_work_group_size;
    err = clGetDeviceInfo(opt->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying max work group size: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = max_work_group_size < 64 ? max_work_group_size : 64;

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = 0.0;
    }

    // Create OpenCL program
    program = clCreateProgramWithSource(opt->context, 1, &bsa_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating BSA program: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building BSA program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    selection_kernel = clCreateKernel(program, "selection_shuffle", &err);
    mutation_kernel = clCreateKernel(program, "mutation", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !selection_kernel || !mutation_kernel || !best_kernel) {
        fprintf(stderr, "Error creating BSA kernels: %d\n", err);
        goto cleanup;
    }

    // Create buffers
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    if (!bounds_float) {
        fprintf(stderr, "Error: Memory allocation failed for bounds_float\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }

    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    historical_pop_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !historical_pop_buffer || !fitness_buffer || !best_idx_buffer) {
        fprintf(stderr, "Error creating BSA buffers: %d\n", err);
        goto cleanup;
    }

    // Write bounds
    err = clEnqueueWriteBuffer(opt->queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bounds buffer: %d\n", err);
        goto cleanup;
    }

    // Initialize population and historical population
    uint seed = (uint)time(NULL);
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 3, sizeof(int), &population_size);
    err |= clSetKernelArg(init_kernel, 4, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting init kernel args for population: %d\n", err);
        goto cleanup;
    }

    size_t init_global_work_size = ((population_size * dim + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(opt->queue, init_kernel, 1, NULL, &init_global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel for population: %d\n", err);
        goto cleanup;
    }

    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &historical_pop_buffer);
    err |= clSetKernelArg(init_kernel, 4, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting init kernel args for historical population: %d\n", err);
        goto cleanup;
    }

    err = clEnqueueNDRangeKernel(opt->queue, init_kernel, 1, NULL, &init_global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel for historical population: %d\n", err);
        goto cleanup;
    }

    // Allocate host buffers
    population = (float*)malloc(population_size * dim * sizeof(float));
    fitness = (float*)malloc(population_size * sizeof(float));
    best_idx_array = (int*)malloc(population_size * sizeof(int));
    if (!population || !fitness || !best_idx_array) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }

    // Main loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Read population for fitness evaluation
        err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                                  population_size * dim * sizeof(float), population, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading population buffer: %d\n", err);
            goto cleanup;
        }

        // Evaluate fitness on host
        #pragma omp parallel for
        for (int i = 0; i < population_size; i++) {
            double* pos_double = (double*)malloc(dim * sizeof(double));
            if (pos_double) {
                for (int d = 0; d < dim; d++) {
                    pos_double[d] = (double)population[i * dim + d];
                }
                fitness[i] = (float)objective_function(pos_double);
                free(pos_double);
            } else {
                fitness[i] = INFINITY;
            }
        }

        // Write fitness to GPU
        err = clEnqueueWriteBuffer(opt->queue, fitness_buffer, CL_TRUE, 0, 
                                   population_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            goto cleanup;
        }

        // Find best solution
        err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &best_idx_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(int), &population_size);
        err |= clSetKernelArg(best_kernel, 3, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 4, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best kernel args: %d\n", err);
            goto cleanup;
        }

        size_t best_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, best_kernel, 1, NULL, &best_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing best kernel: %d\n", err);
            goto cleanup;
        }

        // Update best solution
        int best_idx;
        err = clEnqueueReadBuffer(opt->queue, best_idx_buffer, CL_TRUE, 0, sizeof(int), &best_idx, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best_idx buffer: %d\n", err);
            goto cleanup;
        }
        float best_fitness;
        err = clEnqueueReadBuffer(opt->queue, fitness_buffer, CL_TRUE, best_idx * sizeof(float), sizeof(float), &best_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best fitness: %d\n", err);
            goto cleanup;
        }
        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)best_fitness;
            float* best_solution = (float*)malloc(dim * sizeof(float));
            if (!best_solution) {
                fprintf(stderr, "Error: Memory allocation failed for best_solution\n");
                goto cleanup;
            }
            err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, best_idx * dim * sizeof(float), 
                                      dim * sizeof(float), best_solution, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best solution: %d\n", err);
                free(best_solution);
                goto cleanup;
            }
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)best_solution[d];
            }
            free(best_solution);
        }

        // Selection and pseudo-shuffling
        err = clSetKernelArg(selection_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(selection_kernel, 1, sizeof(cl_mem), &historical_pop_buffer);
        err |= clSetKernelArg(selection_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(selection_kernel, 3, sizeof(int), &population_size);
        err |= clSetKernelArg(selection_kernel, 4, sizeof(uint), &seed);
        err |= clSetKernelArg(selection_kernel, 5, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting selection kernel args: %d\n", err);
            goto cleanup;
        }

        size_t selection_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, selection_kernel, 1, NULL, &selection_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing selection kernel: %d\n", err);
            goto cleanup;
        }

        // Mutation
        err = clSetKernelArg(mutation_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(mutation_kernel, 1, sizeof(cl_mem), &historical_pop_buffer);
        err |= clSetKernelArg(mutation_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(mutation_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(mutation_kernel, 4, sizeof(int), &population_size);
        err |= clSetKernelArg(mutation_kernel, 5, sizeof(uint), &seed);
        err |= clSetKernelArg(mutation_kernel, 6, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting mutation kernel args: %d\n", err);
            goto cleanup;
        }

        size_t mutation_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, mutation_kernel, 1, NULL, &mutation_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing mutation kernel: %d\n", err);
            goto cleanup;
        }

        printf("BSA|%5d -----> %9.16f\n", iter + 1, opt->best_solution.fitness);
    }

    // Final population read
    err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                              population_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final population buffer: %d\n", err);
        goto cleanup;
    }

cleanup:
    if (population) free(population);
    if (fitness) free(fitness);
    if (best_idx_array) free(best_idx_array);
    if (bounds_float) free(bounds_float);
    population = NULL;
    fitness = NULL;
    best_idx_array = NULL;
    bounds_float = NULL;

    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (historical_pop_buffer) clReleaseMemObject(historical_pop_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (selection_kernel) clReleaseKernel(selection_kernel);
    if (mutation_kernel) clReleaseKernel(mutation_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    clFinish(opt->queue);
}
