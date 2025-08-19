/* CBO.c - GPU-Optimized Colliding Bodies Optimization */
#include "CBO.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// OpenCL kernel for CBO
static const char* cbo_kernel_source =
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
"        // Simple LCG random number generator\n"
"        local_seed = local_seed * 1103515245 + 12345;\n"
"        float r = ((local_seed >> 16) & 0x7FFF) / (float)0x7FFF;\n"
"        float min = bounds[2 * d];\n"
"        float max = bounds[2 * d + 1];\n"
"        population[body_idx * dim + d] = min + (max - min) * r;\n"
"    }\n"
"}\n"
"// Compute center of mass (partial sum per dimension)\n"
"__kernel void compute_center_of_mass(\n"
"    __global const float* population,\n"
"    __global float* com_partial,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    __local float* local_sums)\n"
"{\n"
"    int global_id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int group_id = get_group_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    int d = group_id % dim;\n"
"    int block = group_id / dim;\n"
"    local_sums[local_id] = 0.0f;\n"
"    if (global_id < population_size) {\n"
"        local_sums[local_id] = population[global_id * dim + d];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    // Parallel reduction\n"
"    for (int offset = local_size / 2; offset > 0; offset /= 2) {\n"
"        if (local_id < offset) {\n"
"            local_sums[local_id] += local_sums[local_id + offset];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (local_id == 0) {\n"
"        com_partial[group_id] = local_sums[0];\n"
"    }\n"
"}\n"
"// Finalize center of mass\n"
"__kernel void finalize_center_of_mass(\n"
"    __global float* com_partial,\n"
"    __global float* center_of_mass,\n"
"    const int dim,\n"
"    const int num_blocks,\n"
"    const float inv_pop_size)\n"
"{\n"
"    int d = get_global_id(0);\n"
"    if (d < dim) {\n"
"        float sum = 0.0f;\n"
"        for (int b = 0; b < num_blocks; b++) {\n"
"            sum += com_partial[b * dim + d];\n"
"        }\n"
"        center_of_mass[d] = sum * inv_pop_size;\n"
"    }\n"
"}\n"
"// Collision phase\n"
"#define CBO_ALPHA 0.1f\n"
"__kernel void collision_phase(\n"
"    __global float* population,\n"
"    __global const float* center_of_mass,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    __local float* local_com,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < population_size) {\n"
"        // Cache center of mass and bounds in local memory\n"
"        if (local_id < dim) {\n"
"            local_com[local_id] = center_of_mass[local_id];\n"
"        }\n"
"        if (local_id < 2 * dim) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        // Update position\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float pos = population[id * dim + d];\n"
"            float direction = local_com[d] - pos;\n"
"            pos += CBO_ALPHA * direction;\n"
"            // Enforce bounds\n"
"            float lower = local_bounds[2 * d];\n"
"            float upper = local_bounds[2 * d + 1];\n"
"            population[id * dim + d] = clamp(pos, lower, upper);\n"
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

void CBO_optimize(Optimizer* opt, double (*objective_function)(double*)) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, com_kernel = NULL, finalize_com_kernel = NULL;
    cl_kernel collision_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, com_buffer = NULL, com_partial_buffer = NULL;
    cl_mem fitness_buffer = NULL, best_idx_buffer = NULL;
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
    program = clCreateProgramWithSource(opt->context, 1, &cbo_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating CBO program: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building CBO program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    com_kernel = clCreateKernel(program, "compute_center_of_mass", &err);
    finalize_com_kernel = clCreateKernel(program, "finalize_center_of_mass", &err);
    collision_kernel = clCreateKernel(program, "collision_phase", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !com_kernel || !finalize_com_kernel || 
        !collision_kernel || !best_kernel) {
        fprintf(stderr, "Error creating CBO kernels: %d\n", err);
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
    com_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    com_partial_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !com_buffer || !com_partial_buffer || 
        !fitness_buffer || !best_idx_buffer) {
        fprintf(stderr, "Error creating CBO buffers: %d\n", err);
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
    best_idx_array = (int*)malloc(population_size * sizeof(int));
    if (!population || !fitness || !best_idx_array) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }

    // Main loop
    float inv_pop_size = 1.0f / population_size;
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

        // Compute center of mass
        err = clSetKernelArg(com_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(com_kernel, 1, sizeof(cl_mem), &com_partial_buffer);
        err |= clSetKernelArg(com_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(com_kernel, 3, sizeof(int), &population_size);
        err |= clSetKernelArg(com_kernel, 4, local_work_size * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting com kernel args: %d\n", err);
            goto cleanup;
        }

        size_t com_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size * dim;
        err = clEnqueueNDRangeKernel(opt->queue, com_kernel, 1, NULL, &com_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing com kernel: %d\n", err);
            goto cleanup;
        }

        // Finalize center of mass
        err = clSetKernelArg(finalize_com_kernel, 0, sizeof(cl_mem), &com_partial_buffer);
        err |= clSetKernelArg(finalize_com_kernel, 1, sizeof(cl_mem), &com_buffer);
        err |= clSetKernelArg(finalize_com_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(finalize_com_kernel, 3, sizeof(int), &population_size);
        err |= clSetKernelArg(finalize_com_kernel, 4, sizeof(float), &inv_pop_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting finalize_com kernel args: %d\n", err);
            goto cleanup;
        }

        size_t finalize_com_global_work_size = ((dim + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, finalize_com_kernel, 1, NULL, &finalize_com_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing finalize_com kernel: %d\n", err);
            goto cleanup;
        }

        // Collision phase
        err = clSetKernelArg(collision_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(collision_kernel, 1, sizeof(cl_mem), &com_buffer);
        err |= clSetKernelArg(collision_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(collision_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(collision_kernel, 4, sizeof(int), &population_size);
        err |= clSetKernelArg(collision_kernel, 5, dim * sizeof(float), NULL);
        err |= clSetKernelArg(collision_kernel, 6, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting collision kernel args: %d\n", err);
            goto cleanup;
        }

        size_t collision_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, collision_kernel, 1, NULL, &collision_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing collision kernel: %d\n", err);
            goto cleanup;
        }

        printf("CBO|%5d -----> %9.16f\n", iter + 1, opt->best_solution.fitness);
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
    if (com_buffer) clReleaseMemObject(com_buffer);
    if (com_partial_buffer) clReleaseMemObject(com_partial_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (com_kernel) clReleaseKernel(com_kernel);
    if (finalize_com_kernel) clReleaseKernel(finalize_com_kernel);
    if (collision_kernel) clReleaseKernel(collision_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    clFinish(opt->queue);
}
