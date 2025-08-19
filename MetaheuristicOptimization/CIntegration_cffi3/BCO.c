/* BCO.c - GPU-Optimized Bacterial Colony Optimization */
#include "BCO.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// OpenCL kernel for BCO
static const char* bco_kernel_source =
"// Simple LCG random number generator\n"
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
"// Gaussian random number using Box-Muller transform\n"
"float lcg_rand_gaussian(uint* seed) {\n"
"    float u1 = lcg_rand_float(seed, 0.0f, 1.0f);\n"
"    float u2 = lcg_rand_float(seed, 0.0f, 1.0f);\n"
"    return sqrt(-2.0f * log(u1)) * cos(2.0f * 3.14159265359f * u2);\n"
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
"        int bacterium_idx = id / dim;\n"
"        int d = id % dim;\n"
"        uint local_seed = seed + id;\n"
"        float min = bounds[2 * d];\n"
"        float max = bounds[2 * d + 1];\n"
"        population[bacterium_idx * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"    }\n"
"}\n"
"// Chemotaxis and communication phase\n"
"#define BCO_CHEMOTAXIS_STEP_MAX 0.2f\n"
"#define BCO_CHEMOTAXIS_STEP_MIN 0.01f\n"
"#define BCO_COMMUNICATION_PROB 0.5f\n"
"__kernel void chemotaxis_communication(\n"
"    __global float* population,\n"
"    __global const float* best_solution,\n"
"    __global float* fitness,\n"
"    __global float* temp_solutions,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int iteration,\n"
"    const int max_iter,\n"
"    uint seed,\n"
"    __local float* local_best)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id;\n"
"        // Cache best solution in local memory\n"
"        if (local_id == 0) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                local_best[d] = best_solution[d];\n"
"            }\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        // Compute chemotaxis step\n"
"        float chemotaxis_step = BCO_CHEMOTAXIS_STEP_MIN + (BCO_CHEMOTAXIS_STEP_MAX - BCO_CHEMOTAXIS_STEP_MIN) * \n"
"                               ((float)(max_iter - iteration) / max_iter);\n"
"        // Chemotaxis\n"
"        float r = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float temp[32]; // Assumes dim <= 32\n"
"        if (r < 0.5f) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float turbulent = lcg_rand_gaussian(&local_seed);\n"
"                temp[d] = population[id * dim + d] + chemotaxis_step * \n"
"                         (0.5f * (local_best[d] - population[id * dim + d]) + turbulent);\n"
"            }\n"
"        } else {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                temp[d] = population[id * dim + d] + chemotaxis_step * \n"
"                         0.5f * (local_best[d] - population[id * dim + d]);\n"
"            }\n"
"        }\n"
"        // Communication\n"
"        if (lcg_rand_float(&local_seed, 0.0f, 1.0f) < BCO_COMMUNICATION_PROB) {\n"
"            int neighbor_idx = (lcg_rand_float(&local_seed, 0.0f, 1.0f) < 0.5f) ? \n"
"                              (id + (lcg_rand(&local_seed) % 2 == 0 ? -1 : 1) + population_size) % population_size : \n"
"                              lcg_rand(&local_seed) % population_size;\n"
"            if (fitness[neighbor_idx] < fitness[id]) {\n"
"                for (int d = 0; d < dim; d++) {\n"
"                    temp[d] = population[neighbor_idx * dim + d];\n"
"                }\n"
"            } else if (fitness[id] > fitness[0]) { // Best fitness at index 0\n"
"                for (int d = 0; d < dim; d++) {\n"
"                    temp[d] = local_best[d];\n"
"                }\n"
"            }\n"
"        }\n"
"        // Clamp to bounds and store in temp_solutions\n"
"        for (int d = 0; d < dim; d++) {\n"
"            temp_solutions[id * dim + d] = clamp(temp[d], bounds[2 * d], bounds[2 * d + 1]);\n"
"        }\n"
"    }\n"
"}\n"
"// Update population from temp_solutions\n"
"__kernel void update_population(\n"
"    __global float* population,\n"
"    __global const float* temp_solutions,\n"
"    const int dim,\n"
"    const int population_size)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size * dim) {\n"
"        population[id] = temp_solutions[id];\n"
"    }\n"
"}\n"
"// Find best and worst indices\n"
"__kernel void find_best_worst(\n"
"    __global const float* fitness,\n"
"    __global int* best_idx,\n"
"    __global int* worst_indices,\n"
"    const int population_size,\n"
"    const int worst_count,\n"
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
"    // Parallel reduction for best\n"
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
"    // Parallel selection for worst\n"
"    if (id < population_size) {\n"
"        local_fitness[local_id] = -fitness[id]; // Max-heap for worst\n"
"    } else {\n"
"        local_fitness[local_id] = -INFINITY;\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int i = 0; i < worst_count; i++) {\n"
"        for (int offset = local_size / 2; offset > 0; offset /= 2) {\n"
"            if (local_id < offset) {\n"
"                if (local_fitness[local_id + offset] > local_fitness[local_id]) {\n"
"                    local_fitness[local_id] = local_fitness[local_id + offset];\n"
"                    local_indices[local_id] = local_indices[local_id + offset];\n"
"                }\n"
"            }\n"
"            barrier(CLK_LOCAL_MEM_FENCE);\n"
"        }\n"
"        if (local_id == 0 && i < worst_count) {\n"
"            worst_indices[i] = local_indices[0];\n"
"            local_fitness[0] = -INFINITY;\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"}\n"
"// Elimination, reproduction, and migration phase\n"
"#define BCO_ELIMINATION_RATIO 0.2f\n"
"#define BCO_REPRODUCTION_THRESHOLD 0.5f\n"
"#define BCO_MIGRATION_PROBABILITY 0.1f\n"
"__kernel void elimination_reproduction_migration(\n"
"    __global float* population,\n"
"    __global float* fitness,\n"
"    __global const float* best_solution,\n"
"    __global const float* bounds,\n"
"    __global const int* worst_indices,\n"
"    __global const int* best_idx,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int worst_count,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < worst_count) {\n"
"        uint local_seed = seed + id;\n"
"        int idx = worst_indices[id];\n"
"        // Compute energy\n"
"        float energy = 1.0f / (1.0f + fitness[idx]);\n"
"        // Elimination\n"
"        if (energy < BCO_REPRODUCTION_THRESHOLD) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float min = bounds[2 * d];\n"
"                float max = bounds[2 * d + 1];\n"
"                population[idx * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"            }\n"
"            fitness[idx] = INFINITY;\n"
"        }\n"
"        // Reproduction\n"
"        if (id < worst_count / 2) {\n"
"            int src_idx = best_idx[0]; // Use best bacterium\n"
"            int dst_idx = worst_indices[worst_count - 1 - id];\n"
"            if (1.0f / (1.0f + fitness[src_idx]) >= BCO_REPRODUCTION_THRESHOLD) {\n"
"                for (int d = 0; d < dim; d++) {\n"
"                    population[dst_idx * dim + d] = population[src_idx * dim + d];\n"
"                }\n"
"                fitness[dst_idx] = fitness[src_idx];\n"
"            }\n"
"        }\n"
"    } else if (id < population_size) {\n"
"        uint local_seed = seed + id;\n"
"        // Migration\n"
"        if (lcg_rand_float(&local_seed, 0.0f, 1.0f) < BCO_MIGRATION_PROBABILITY) {\n"
"            float energy = 1.0f / (1.0f + fitness[id]);\n"
"            float norm = 0.0f;\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float diff = population[id * dim + d] - best_solution[d];\n"
"                norm += diff * diff;\n"
"            }\n"
"            norm = sqrt(norm);\n"
"            if (energy < BCO_REPRODUCTION_THRESHOLD || norm < 1e-3f) {\n"
"                for (int d = 0; d < dim; d++) {\n"
"                    float min = bounds[2 * d];\n"
"                    float max = bounds[2 * d + 1];\n"
"                    population[id * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"                }\n"
"                fitness[id] = INFINITY;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n";

void BCO_optimize(Optimizer* opt, double (*objective_function)(double*)) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, chemotaxis_kernel = NULL, update_kernel = NULL;
    cl_kernel best_worst_kernel = NULL, elim_repro_migration_kernel = NULL;
    cl_mem bounds_buffer = NULL, best_solution_buffer = NULL, temp_buffer = NULL;
    cl_mem fitness_buffer = NULL, best_idx_buffer = NULL, worst_indices_buffer = NULL;
    float* bounds_float = NULL, *population = NULL, *fitness = NULL, *best_solution = NULL;
    int* worst_indices = NULL;

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
    const int worst_count = (int)(BCO_ELIMINATION_RATIO * population_size);
    if (dim < 1 || population_size < 1 || max_iter < 1 || worst_count < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), max_iter (%d), or worst_count (%d)\n", 
                dim, population_size, max_iter, worst_count);
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
    program = clCreateProgramWithSource(opt->context, 1, &bco_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating BCO program: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building BCO program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    chemotaxis_kernel = clCreateKernel(program, "chemotaxis_communication", &err);
    update_kernel = clCreateKernel(program, "update_population", &err);
    best_worst_kernel = clCreateKernel(program, "find_best_worst", &err);
    elim_repro_migration_kernel = clCreateKernel(program, "elimination_reproduction_migration", &err);
    if (err != CL_SUCCESS || !init_kernel || !chemotaxis_kernel || !update_kernel || 
        !best_worst_kernel || !elim_repro_migration_kernel) {
        fprintf(stderr, "Error creating BCO kernels: %d\n", err);
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
    best_solution_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    temp_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    worst_indices_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, worst_count * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !best_solution_buffer || !temp_buffer || 
        !fitness_buffer || !best_idx_buffer || !worst_indices_buffer) {
        fprintf(stderr, "Error creating BCO buffers: %d\n", err);
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
    best_solution = (float*)malloc(dim * sizeof(float));
    worst_indices = (int*)malloc(worst_count * sizeof(int));
    if (!population || !fitness || !best_solution || !worst_indices) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }

    // Main loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Read population for fitness evaluation (host-side)
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

        // Find best and worst indices
        err = clSetKernelArg(best_worst_kernel, 0, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(best_worst_kernel, 1, sizeof(cl_mem), &best_idx_buffer);
        err |= clSetKernelArg(best_worst_kernel, 2, sizeof(cl_mem), &worst_indices_buffer);
        err |= clSetKernelArg(best_worst_kernel, 3, sizeof(int), &population_size);
        err |= clSetKernelArg(best_worst_kernel, 4, sizeof(int), &worst_count);
        err |= clSetKernelArg(best_worst_kernel, 5, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_worst_kernel, 6, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best_worst kernel args: %d\n", err);
            goto cleanup;
        }

        size_t best_worst_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, best_worst_kernel, 1, NULL, &best_worst_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing best_worst kernel: %d\n", err);
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
            err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, best_idx * dim * sizeof(float), 
                                      dim * sizeof(float), best_solution, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best solution: %d\n", err);
                goto cleanup;
            }
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)best_solution[d];
            }
            err = clEnqueueWriteBuffer(opt->queue, best_solution_buffer, CL_TRUE, 0, dim * sizeof(float), best_solution, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing best_solution buffer: %d\n", err);
                goto cleanup;
            }
        }

        // Chemotaxis and communication
        err = clSetKernelArg(chemotaxis_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(chemotaxis_kernel, 1, sizeof(cl_mem), &best_solution_buffer);
        err |= clSetKernelArg(chemotaxis_kernel, 2, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(chemotaxis_kernel, 3, sizeof(cl_mem), &temp_buffer);
        err |= clSetKernelArg(chemotaxis_kernel, 4, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(chemotaxis_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(chemotaxis_kernel, 6, sizeof(int), &population_size);
        err |= clSetKernelArg(chemotaxis_kernel, 7, sizeof(int), &iter);
        err |= clSetKernelArg(chemotaxis_kernel, 8, sizeof(int), &max_iter);
        err |= clSetKernelArg(chemotaxis_kernel, 9, sizeof(uint), &seed);
        err |= clSetKernelArg(chemotaxis_kernel, 10, dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting chemotaxis kernel args: %d\n", err);
            goto cleanup;
        }

        size_t chemotaxis_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, chemotaxis_kernel, 1, NULL, &chemotaxis_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing chemotaxis kernel: %d\n", err);
            goto cleanup;
        }

        // Update population
        err = clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &temp_buffer);
        err |= clSetKernelArg(update_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(update_kernel, 3, sizeof(int), &population_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting update kernel args: %d\n", err);
            goto cleanup;
        }

        size_t update_global_work_size = ((population_size * dim + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, update_kernel, 1, NULL, &update_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing update kernel: %d\n", err);
            goto cleanup;
        }

        // Elimination, reproduction, and migration
        err = clSetKernelArg(elim_repro_migration_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(elim_repro_migration_kernel, 1, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(elim_repro_migration_kernel, 2, sizeof(cl_mem), &best_solution_buffer);
        err |= clSetKernelArg(elim_repro_migration_kernel, 3, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(elim_repro_migration_kernel, 4, sizeof(cl_mem), &worst_indices_buffer);
        err |= clSetKernelArg(elim_repro_migration_kernel, 5, sizeof(cl_mem), &best_idx_buffer);
        err |= clSetKernelArg(elim_repro_migration_kernel, 6, sizeof(int), &dim);
        err |= clSetKernelArg(elim_repro_migration_kernel, 7, sizeof(int), &population_size);
        err |= clSetKernelArg(elim_repro_migration_kernel, 8, sizeof(int), &worst_count);
        err |= clSetKernelArg(elim_repro_migration_kernel, 9, sizeof(uint), &seed);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting elim_repro_migration kernel args: %d\n", err);
            goto cleanup;
        }

        size_t elim_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, elim_repro_migration_kernel, 1, NULL, &elim_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing elim_repro_migration kernel: %d\n", err);
            goto cleanup;
        }

        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Final best solution
    err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                              population_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final population buffer: %d\n", err);
        goto cleanup;
    }

cleanup:
    if (population) free(population);
    if (fitness) free(fitness);
    if (best_solution) free(best_solution);
    if (worst_indices) free(worst_indices);
    if (bounds_float) free(bounds_float);
    population = NULL;
    fitness = NULL;
    best_solution = NULL;
    worst_indices = NULL;
    bounds_float = NULL;

    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (best_solution_buffer) clReleaseMemObject(best_solution_buffer);
    if (temp_buffer) clReleaseMemObject(temp_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (worst_indices_buffer) clReleaseMemObject(worst_indices_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (chemotaxis_kernel) clReleaseKernel(chemotaxis_kernel);
    if (update_kernel) clReleaseKernel(update_kernel);
    if (best_worst_kernel) clReleaseKernel(best_worst_kernel);
    if (elim_repro_migration_kernel) clReleaseKernel(elim_repro_migration_kernel);
    if (program) clReleaseProgram(program);
    clFinish(opt->queue);
}
