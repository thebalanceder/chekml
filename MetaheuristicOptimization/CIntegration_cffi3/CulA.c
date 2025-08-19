#include "CulA.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// OpenCL kernel source for Cultural Algorithm
static const char* cula_kernel_source =
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
"// Schwefel function for fitness evaluation\n"
"float schwefel(__global const float* position, const int dim) {\n"
"    float sum = 0.0f;\n"
"    for (int i = 0; i < dim; i++) {\n"
"        float x = position[i];\n"
"        sum += x * sin(sqrt(fabs(x)));\n"
"    }\n"
"    return 418.9829f * dim - sum;\n"
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
"        int idx = id / dim;\n"
"        int d = id % dim;\n"
"        uint local_seed = seed + id;\n"
"        float min = bounds[2 * d];\n"
"        float max = bounds[2 * d + 1];\n"
"        population[idx * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"    }\n"
"}\n"
"// Evaluate fitness\n"
"__kernel void evaluate_fitness(\n"
"    __global const float* population,\n"
"    __global float* fitness,\n"
"    const int dim,\n"
"    const int population_size)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        fitness[id] = schwefel(&population[id * dim], dim);\n"
"    }\n"
"}\n"
"// Find top n_accept indices (parallel selection)\n"
"__kernel void find_top_n_accept(\n"
"    __global const float* fitness,\n"
"    __global int* indices,\n"
"    const int population_size,\n"
"    const int n_accept,\n"
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
"    // Parallel selection for top n_accept\n"
"    for (int i = 0; i < n_accept; i++) {\n"
"        for (int offset = local_size / 2; offset > 0; offset /= 2) {\n"
"            if (local_id < offset) {\n"
"                if (local_fitness[local_id + offset] < local_fitness[local_id]) {\n"
"                    local_fitness[local_id] = local_fitness[local_id + offset];\n"
"                    local_indices[local_id] = local_indices[local_id + offset];\n"
"                }\n"
"            }\n"
"            barrier(CLK_LOCAL_MEM_FENCE);\n"
"        }\n"
"        if (local_id == 0 && i < n_accept) {\n"
"            indices[i] = local_indices[0];\n"
"            local_fitness[0] = INFINITY;\n"
"            local_indices[0] = -1;\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"}\n"
"// Adjust culture based on top n_accept individuals\n"
"#define ACCEPTANCE_RATIO 0.2f\n"
"__kernel void adjust_culture(\n"
"    __global const float* population,\n"
"    __global const float* fitness,\n"
"    __global const int* indices,\n"
"    __global float* situational_position,\n"
"    __global float* situational_cost,\n"
"    __global float* normative_min,\n"
"    __global float* normative_max,\n"
"    __global float* normative_L,\n"
"    __global float* normative_U,\n"
"    __global float* normative_size,\n"
"    const int dim,\n"
"    const int n_accept)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < n_accept) {\n"
"        int idx = indices[id];\n"
"        float cost = fitness[idx];\n"
"        // Update situational knowledge\n"
"        if (id == 0) {\n"
"            if (cost < *situational_cost) {\n"
"                *situational_cost = cost;\n"
"                for (int d = 0; d < dim; d++) {\n"
"                    situational_position[d] = population[idx * dim + d];\n"
"                }\n"
"            }\n"
"        }\n"
"        // Update normative knowledge\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float pos = population[idx * dim + d];\n"
"            if (pos < normative_min[d] || cost < normative_L[d]) {\n"
"                normative_min[d] = pos;\n"
"                normative_L[d] = cost;\n"
"            }\n"
"            if (pos > normative_max[d] || cost < normative_U[d]) {\n"
"                normative_max[d] = pos;\n"
"                normative_U[d] = cost;\n"
"            }\n"
"            normative_size[d] = normative_max[d] - normative_min[d];\n"
"        }\n"
"    }\n"
"}\n"
"// Apply cultural influence\n"
"#define ALPHA_SCALING 0.1f\n"
"__kernel void influence_culture(\n"
"    __global float* population,\n"
"    __global const float* situational_position,\n"
"    __global const float* normative_size,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed,\n"
"    __local float* local_situational)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id;\n"
"        // Cache situational position in local memory\n"
"        if (local_id == 0) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                local_situational[d] = situational_position[d];\n"
"            }\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        // Apply influence\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float sigma = ALPHA_SCALING * normative_size[d];\n"
"            float dx = lcg_rand_gaussian(&local_seed) * sigma;\n"
"            float pos = population[id * dim + d];\n"
"            float situational = local_situational[d];\n"
"            if (pos < situational) {\n"
"                dx = fabs(dx);\n"
"            } else if (pos > situational) {\n"
"                dx = -fabs(dx);\n"
"            }\n"
"            float new_pos = pos + dx;\n"
"            // Clamp to bounds\n"
"            population[id * dim + d] = clamp(new_pos, bounds[2 * d], bounds[2 * d + 1]);\n"
"        }\n"
"    }\n"
"}\n";

// Main Optimization Function
void CulA_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, fitness_kernel = NULL, top_n_kernel = NULL;
    cl_kernel adjust_kernel = NULL, influence_kernel = NULL;
    cl_mem bounds_buffer = NULL, indices_buffer = NULL;
    cl_mem situational_position_buffer = NULL, situational_cost_buffer = NULL;
    cl_mem normative_min_buffer = NULL, normative_max_buffer = NULL;
    cl_mem normative_L_buffer = NULL, normative_U_buffer = NULL;
    cl_mem normative_size_buffer = NULL;
    float *bounds_float = NULL, *situational_position = NULL;
    float *fitness = NULL;
    int *indices = NULL;
    Culture culture = {0};

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
    const int n_accept = (int)(ACCEPTANCE_RATIO * population_size);
    if (dim < 1 || population_size < 1 || max_iter < 1 || n_accept < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), max_iter (%d), or n_accept (%d)\n", 
                dim, population_size, max_iter, n_accept);
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
    program = clCreateProgramWithSource(opt->context, 1, &cula_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating CulA program: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building CulA program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating init_kernel: %d\n", err); goto cleanup; }
    fitness_kernel = clCreateKernel(program, "evaluate_fitness", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating fitness_kernel: %d\n", err); goto cleanup; }
    top_n_kernel = clCreateKernel(program, "find_top_n_accept", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating top_n_kernel: %d\n", err); goto cleanup; }
    adjust_kernel = clCreateKernel(program, "adjust_culture", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating adjust_kernel: %d\n", err); goto cleanup; }
    influence_kernel = clCreateKernel(program, "influence_culture", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating influence_kernel: %d\n", err); goto cleanup; }

    // Create buffers
    bounds_float = (float *)malloc(2 * dim * sizeof(float));
    if (!bounds_float) { fprintf(stderr, "Error allocating bounds_float\n"); goto cleanup; }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }

    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating bounds_buffer: %d\n", err); goto cleanup; }
    indices_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, n_accept * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating indices_buffer: %d\n", err); goto cleanup; }
    situational_position_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating situational_position_buffer: %d\n", err); goto cleanup; }
    situational_cost_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating situational_cost_buffer: %d\n", err); goto cleanup; }
    normative_min_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating normative_min_buffer: %d\n", err); goto cleanup; }
    normative_max_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating normative_max_buffer: %d\n", err); goto cleanup; }
    normative_L_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating normative_L_buffer: %d\n", err); goto cleanup; }
    normative_U_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating normative_U_buffer: %d\n", err); goto cleanup; }
    normative_size_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating normative_size_buffer: %d\n", err); goto cleanup; }

    // Write bounds
    err = clEnqueueWriteBuffer(opt->queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error writing bounds_buffer: %d\n", err); goto cleanup; }

    // Initialize culture buffers
    float init_cost = INFINITY;
    float *init_norm = (float *)malloc(dim * sizeof(float));
    if (!init_norm) { fprintf(stderr, "Error allocating init_norm\n"); goto cleanup; }
    for (int i = 0; i < dim; i++) {
        init_norm[i] = INFINITY;
    }
    err = clEnqueueWriteBuffer(opt->queue, situational_cost_buffer, CL_TRUE, 0, sizeof(float), &init_cost, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error writing situational_cost_buffer: %d\n", err); goto cleanup; }
    err = clEnqueueWriteBuffer(opt->queue, normative_min_buffer, CL_TRUE, 0, dim * sizeof(float), init_norm, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error writing normative_min_buffer: %d\n", err); goto cleanup; }
    err = clEnqueueWriteBuffer(opt->queue, normative_max_buffer, CL_TRUE, 0, dim * sizeof(float), init_norm, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error writing normative_max_buffer: %d\n", err); goto cleanup; }
    err = clEnqueueWriteBuffer(opt->queue, normative_L_buffer, CL_TRUE, 0, dim * sizeof(float), init_norm, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error writing normative_L_buffer: %d\n", err); goto cleanup; }
    err = clEnqueueWriteBuffer(opt->queue, normative_U_buffer, CL_TRUE, 0, dim * sizeof(float), init_norm, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error writing normative_U_buffer: %d\n", err); goto cleanup; }
    for (int i = 0; i < dim; i++) {
        init_norm[i] = 0.0f;
    }
    err = clEnqueueWriteBuffer(opt->queue, normative_size_buffer, CL_TRUE, 0, dim * sizeof(float), init_norm, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error writing normative_size_buffer: %d\n", err); goto cleanup; }
    free(init_norm);
    init_norm = NULL;

    // Initialize population
    uint seed = (uint)time(NULL);
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 3, sizeof(int), &population_size);
    err |= clSetKernelArg(init_kernel, 4, sizeof(uint), &seed);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error setting init_kernel args: %d\n", err); goto cleanup; }

    size_t init_global_work_size = ((population_size * dim + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(opt->queue, init_kernel, 1, NULL, &init_global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error enqueuing init_kernel: %d\n", err); goto cleanup; }

    // Allocate host buffers
    situational_position = (float *)malloc(dim * sizeof(float));
    if (!situational_position) { fprintf(stderr, "Error allocating situational_position\n"); goto cleanup; }
    fitness = (float *)malloc(population_size * sizeof(float));
    if (!fitness) { fprintf(stderr, "Error allocating fitness\n"); goto cleanup; }
    indices = (int *)malloc(n_accept * sizeof(int));
    if (!indices) { fprintf(stderr, "Error allocating indices\n"); goto cleanup; }

    // Main loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Evaluate fitness
        err = clSetKernelArg(fitness_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(fitness_kernel, 1, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(fitness_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(fitness_kernel, 3, sizeof(int), &population_size);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error setting fitness_kernel args: %d\n", err); goto cleanup; }

        size_t fitness_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, fitness_kernel, 1, NULL, &fitness_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error enqueuing fitness_kernel: %d\n", err); goto cleanup; }

        // Find top n_accept indices
        err = clSetKernelArg(top_n_kernel, 0, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(top_n_kernel, 1, sizeof(cl_mem), &indices_buffer);
        err |= clSetKernelArg(top_n_kernel, 2, sizeof(int), &population_size);
        err |= clSetKernelArg(top_n_kernel, 3, sizeof(int), &n_accept);
        err |= clSetKernelArg(top_n_kernel, 4, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(top_n_kernel, 5, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error setting top_n_kernel args: %d\n", err); goto cleanup; }

        size_t top_n_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, top_n_kernel, 1, NULL, &top_n_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error enqueuing top_n_kernel: %d\n", err); goto cleanup; }

        // Adjust culture
        err = clSetKernelArg(adjust_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(adjust_kernel, 1, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(adjust_kernel, 2, sizeof(cl_mem), &indices_buffer);
        err |= clSetKernelArg(adjust_kernel, 3, sizeof(cl_mem), &situational_position_buffer);
        err |= clSetKernelArg(adjust_kernel, 4, sizeof(cl_mem), &situational_cost_buffer);
        err |= clSetKernelArg(adjust_kernel, 5, sizeof(cl_mem), &normative_min_buffer);
        err |= clSetKernelArg(adjust_kernel, 6, sizeof(cl_mem), &normative_max_buffer);
        err |= clSetKernelArg(adjust_kernel, 7, sizeof(cl_mem), &normative_L_buffer);
        err |= clSetKernelArg(adjust_kernel, 8, sizeof(cl_mem), &normative_U_buffer);
        err |= clSetKernelArg(adjust_kernel, 9, sizeof(cl_mem), &normative_size_buffer);
        err |= clSetKernelArg(adjust_kernel, 10, sizeof(int), &dim);
        err |= clSetKernelArg(adjust_kernel, 11, sizeof(int), &n_accept);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error setting adjust_kernel args: %d\n", err); goto cleanup; }

        size_t adjust_global_work_size = ((n_accept + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, adjust_kernel, 1, NULL, &adjust_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error enqueuing adjust_kernel: %d\n", err); goto cleanup; }

        // Influence culture
        err = clSetKernelArg(influence_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(influence_kernel, 1, sizeof(cl_mem), &situational_position_buffer);
        err |= clSetKernelArg(influence_kernel, 2, sizeof(cl_mem), &normative_size_buffer);
        err |= clSetKernelArg(influence_kernel, 3, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(influence_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(influence_kernel, 5, sizeof(int), &population_size);
        err |= clSetKernelArg(influence_kernel, 6, sizeof(uint), &seed);
        err |= clSetKernelArg(influence_kernel, 7, dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error setting influence_kernel args: %d\n", err); goto cleanup; }

        size_t influence_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, influence_kernel, 1, NULL, &influence_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error enqueuing influence_kernel: %d\n", err); goto cleanup; }

        // Update best solution
        float situational_cost;
        err = clEnqueueReadBuffer(opt->queue, situational_cost_buffer, CL_TRUE, 0, sizeof(float), &situational_cost, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error reading situational_cost_buffer: %d\n", err); goto cleanup; }
        if (situational_cost < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)situational_cost;
            err = clEnqueueReadBuffer(opt->queue, situational_position_buffer, CL_TRUE, 0, dim * sizeof(float), situational_position, 0, NULL, NULL);
            if (err != CL_SUCCESS) { fprintf(stderr, "Error reading situational_position_buffer: %d\n", err); goto cleanup; }
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)situational_position[d];
            }
        }

        printf("Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Final best solution
    float situational_cost;
    err = clEnqueueReadBuffer(opt->queue, situational_cost_buffer, CL_TRUE, 0, sizeof(float), &situational_cost, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error reading final situational_cost_buffer: %d\n", err); goto cleanup; }
    if (situational_cost < opt->best_solution.fitness) {
        opt->best_solution.fitness = (double)situational_cost;
        err = clEnqueueReadBuffer(opt->queue, situational_position_buffer, CL_TRUE, 0, dim * sizeof(float), situational_position, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error reading final situational_position_buffer: %d\n", err); goto cleanup; }
        for (int d = 0; d < dim; d++) {
            opt->best_solution.position[d] = (double)situational_position[d];
        }
    }

cleanup:
    if (bounds_float) free(bounds_float);
    if (situational_position) free(situational_position);
    if (fitness) free(fitness);
    if (indices) free(indices);
    if (init_norm) free(init_norm);
    bounds_float = NULL;
    situational_position = NULL;
    fitness = NULL;
    indices = NULL;
    init_norm = NULL;

    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (indices_buffer) clReleaseMemObject(indices_buffer);
    if (situational_position_buffer) clReleaseMemObject(situational_position_buffer);
    if (situational_cost_buffer) clReleaseMemObject(situational_cost_buffer);
    if (normative_min_buffer) clReleaseMemObject(normative_min_buffer);
    if (normative_max_buffer) clReleaseMemObject(normative_max_buffer);
    if (normative_L_buffer) clReleaseMemObject(normative_L_buffer);
    if (normative_U_buffer) clReleaseMemObject(normative_U_buffer);
    if (normative_size_buffer) clReleaseMemObject(normative_size_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (fitness_kernel) clReleaseKernel(fitness_kernel);
    if (top_n_kernel) clReleaseKernel(top_n_kernel);
    if (adjust_kernel) clReleaseKernel(adjust_kernel);
    if (influence_kernel) clReleaseKernel(influence_kernel);
    if (program) clReleaseProgram(program);
    bounds_buffer = NULL;
    indices_buffer = NULL;
    situational_position_buffer = NULL;
    situational_cost_buffer = NULL;
    normative_min_buffer = NULL;
    normative_max_buffer = NULL;
    normative_L_buffer = NULL;
    normative_U_buffer = NULL;
    normative_size_buffer = NULL;
    init_kernel = NULL;
    fitness_kernel = NULL;
    top_n_kernel = NULL;
    adjust_kernel = NULL;
    influence_kernel = NULL;
    program = NULL;

    clFinish(opt->queue);
}
