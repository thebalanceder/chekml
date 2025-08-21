/* CRO.c - GPU-Optimized Coral Reef Optimization */
#include "CRO.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// OpenCL kernel for CRO
static const char* cro_kernel_source =
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
"        int body_idx = id / dim;\n"
"        int d = id % dim;\n"
"        uint local_seed = seed + id;\n"
"        float min = bounds[2 * d];\n"
"        float max = bounds[2 * d + 1];\n"
"        population[body_idx * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"    }\n"
"}\n"
"// Migration phase\n"
"#define CRO_NUM_REEFS 10\n"
"__kernel void migration_phase(\n"
"    __global float* population,\n"
"    __global int* modified,\n"
"    __global int* modified_count,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int solutions_per_reef,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < CRO_NUM_REEFS) {\n"
"        uint local_seed = seed + id;\n"
"        int i = (int)(lcg_rand_float(&local_seed, 0.0f, CRO_NUM_REEFS));\n"
"        int j = (int)(lcg_rand_float(&local_seed, 0.0f, CRO_NUM_REEFS));\n"
"        if (i != j) {\n"
"            int idx = i * solutions_per_reef + (int)(lcg_rand_float(&local_seed, 0.0f, solutions_per_reef));\n"
"            int idx_replace = j * solutions_per_reef + (int)(lcg_rand_float(&local_seed, 0.0f, solutions_per_reef));\n"
"            if (idx < population_size && idx_replace < population_size) {\n"
"                for (int d = 0; d < dim; d++) {\n"
"                    population[idx_replace * dim + d] = population[idx * dim + d];\n"
"                }\n"
"                int pos = atomic_inc(modified_count);\n"
"                if (pos < population_size) {\n"
"                    modified[pos] = idx_replace;\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"// Local search phase\n"
"#define CRO_ALPHA 0.1f\n"
"__kernel void local_search_phase(\n"
"    __global float* population,\n"
"    __global const float* bounds,\n"
"    __global int* modified,\n"
"    __global int* modified_count,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int solutions_per_reef,\n"
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
"        // Perturb position\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float pos = population[id * dim + d];\n"
"            pos += CRO_ALPHA * (2.0f * lcg_rand_float(&local_seed, 0.0f, 1.0f) - 1.0f);\n"
"            // Enforce bounds\n"
"            float lower = local_bounds[2 * d];\n"
"            float upper = local_bounds[2 * d + 1];\n"
"            population[id * dim + d] = clamp(pos, lower, upper);\n"
"        }\n"
"        // Add to modified list\n"
"        int pos = atomic_inc(modified_count);\n"
"        if (pos < population_size) {\n"
"            modified[pos] = id;\n"
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

void CRO_optimize(Optimizer* opt, double (*objective_function)(double*)) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, migration_kernel = NULL, local_search_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, fitness_buffer = NULL, modified_buffer = NULL, modified_count_buffer = NULL;
    cl_mem best_idx_buffer = NULL;
    float* bounds_float = NULL, *population = NULL, *fitness = NULL;
    int* modified = NULL, *best_idx_array = NULL;

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
    if (population_size % CRO_NUM_REEFS != 0) {
        fprintf(stderr, "Error: population_size (%d) must be divisible by CRO_NUM_REEFS (%d)\n", 
                population_size, CRO_NUM_REEFS);
        exit(EXIT_FAILURE);
    }

    const int solutions_per_reef = population_size / CRO_NUM_REEFS;

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
    program = clCreateProgramWithSource(opt->context, 1, &cro_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating CRO program: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building CRO program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    migration_kernel = clCreateKernel(program, "migration_phase", &err);
    local_search_kernel = clCreateKernel(program, "local_search_phase", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !migration_kernel || !local_search_kernel || !best_kernel) {
        fprintf(stderr, "Error creating CRO kernels: %d\n", err);
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
    fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    modified_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    modified_count_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
    best_idx_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !fitness_buffer || !modified_buffer || 
        !modified_count_buffer || !best_idx_buffer) {
        fprintf(stderr, "Error creating CRO buffers: %d\n", err);
        goto cleanup;
    }

    // Write bounds
    err = clEnqueueWriteBuffer(opt->queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bounds buffer: %d\n", err);
        goto cleanup;
    }

    // Initialize population
    uint seed = (uint)time(NULL);
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 3, sizeof(int), &population_size);
    err |= clSetKernelArg(init_kernel, 4, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting init kernel args: %d\n", err);
        goto cleanup;
    }

    size_t init_global_work_size = ((population_size * dim + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(opt->queue, init_kernel, 1, NULL, &init_global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel: %d\n", err);
        goto cleanup;
    }

    // Allocate host buffers
    population = (float*)malloc(population_size * dim * sizeof(float));
    fitness = (float*)malloc(population_size * sizeof(float));
    modified = (int*)malloc(population_size * sizeof(int));
    best_idx_array = (int*)malloc(population_size * sizeof(int));
    if (!population || !fitness || !modified || !best_idx_array) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }

    // Initialize fitness buffer
    for (int i = 0; i < population_size; i++) {
        fitness[i] = INFINITY;
    }
    err = clEnqueueWriteBuffer(opt->queue, fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial fitness buffer: %d\n", err);
        goto cleanup;
    }

    // Evaluate initial population
    err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                              population_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading initial population buffer: %d\n", err);
        goto cleanup;
    }

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

    err = clEnqueueWriteBuffer(opt->queue, fitness_buffer, CL_TRUE, 0, 
                               population_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial fitness buffer: %d\n", err);
        goto cleanup;
    }

    // Main loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Reset modified count
        int zero = 0;
        err = clEnqueueWriteBuffer(opt->queue, modified_count_buffer, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error resetting modified_count buffer: %d\n", err);
            goto cleanup;
        }

        // Migration phase
        err = clSetKernelArg(migration_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(migration_kernel, 1, sizeof(cl_mem), &modified_buffer);
        err |= clSetKernelArg(migration_kernel, 2, sizeof(cl_mem), &modified_count_buffer);
        err |= clSetKernelArg(migration_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(migration_kernel, 4, sizeof(int), &population_size);
        err |= clSetKernelArg(migration_kernel, 5, sizeof(int), &solutions_per_reef);
        err |= clSetKernelArg(migration_kernel, 6, sizeof(uint), &seed);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting migration kernel args: %d\n", err);
            goto cleanup;
        }

        size_t migration_global_work_size = ((CRO_NUM_REEFS + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, migration_kernel, 1, NULL, &migration_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing migration kernel: %d\n", err);
            goto cleanup;
        }

        // Local search phase
        err = clSetKernelArg(local_search_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(local_search_kernel, 1, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(local_search_kernel, 2, sizeof(cl_mem), &modified_buffer);
        err |= clSetKernelArg(local_search_kernel, 3, sizeof(cl_mem), &modified_count_buffer);
        err |= clSetKernelArg(local_search_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(local_search_kernel, 5, sizeof(int), &population_size);
        err |= clSetKernelArg(local_search_kernel, 6, sizeof(int), &solutions_per_reef);
        err |= clSetKernelArg(local_search_kernel, 7, sizeof(uint), &seed);
        err |= clSetKernelArg(local_search_kernel, 8, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting local search kernel args: %d\n", err);
            goto cleanup;
        }

        size_t local_search_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, local_search_kernel, 1, NULL, &local_search_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing local search kernel: %d\n", err);
            goto cleanup;
        }

        // Read modified solutions
        int modified_count;
        err = clEnqueueReadBuffer(opt->queue, modified_count_buffer, CL_TRUE, 0, sizeof(int), &modified_count, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading modified_count buffer: %d\n", err);
            goto cleanup;
        }

        if (modified_count > 0) {
            if (modified_count > population_size) {
                modified_count = population_size;
            }
            err = clEnqueueReadBuffer(opt->queue, modified_buffer, CL_TRUE, 0, modified_count * sizeof(int), modified, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading modified buffer: %d\n", err);
                goto cleanup;
            }

            // Read population for fitness evaluation
            err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                                      population_size * dim * sizeof(float), population, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading population buffer: %d\n", err);
                goto cleanup;
            }

            // Evaluate modified solutions
            #pragma omp parallel for
            for (int m = 0; m < modified_count; m++) {
                int idx = modified[m];
                if (idx >= 0 && idx < population_size) {
                    double* pos_double = (double*)malloc(dim * sizeof(double));
                    if (pos_double) {
                        for (int d = 0; d < dim; d++) {
                            pos_double[d] = (double)population[idx * dim + d];
                        }
                        fitness[idx] = (float)objective_function(pos_double);
                        free(pos_double);
                    } else {
                        fitness[idx] = INFINITY;
                    }
                }
            }

            // Write updated fitness
            err = clEnqueueWriteBuffer(opt->queue, fitness_buffer, CL_TRUE, 0, 
                                       population_size * sizeof(float), fitness, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing fitness buffer: %d\n", err);
                goto cleanup;
            }
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

        printf("CRO|%5d -----> %9.16f\n", iter + 1, opt->best_solution.fitness);
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
    if (modified) free(modified);
    if (best_idx_array) free(best_idx_array);
    if (bounds_float) free(bounds_float);
    population = NULL;
    fitness = NULL;
    modified = NULL;
    best_idx_array = NULL;
    bounds_float = NULL;

    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (modified_buffer) clReleaseMemObject(modified_buffer);
    if (modified_count_buffer) clReleaseMemObject(modified_count_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (migration_kernel) clReleaseKernel(migration_kernel);
    if (local_search_kernel) clReleaseKernel(local_search_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    clFinish(opt->queue);
}
