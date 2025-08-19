#include "DE.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Define string representation of DE_STRATEGY
#if DE_STRATEGY == 0
#define DE_STRATEGY_STR "0"
#elif DE_STRATEGY == 1
#define DE_STRATEGY_STR "1"
#elif DE_STRATEGY == 2
#define DE_STRATEGY_STR "2"
#else
#error "Invalid DE_STRATEGY value in DE.h (must be 0, 1, or 2)"
#endif

// OpenCL kernel source for Differential Evolution
static const char* de_kernel_source =
"// Linear congruential generator\n"
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed) {\n"
"    return ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
"int lcg_rand_int(uint* seed, int max) {\n"
"    return lcg_rand(seed) % max;\n"
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
"    if (id < population_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            population[id * dim + d] = min + (max - min) * lcg_rand_float(&local_seed);\n"
"        }\n"
"    }\n"
"}\n"
"// Mutate and crossover\n"
"#define DE_F 0.5f\n"
"#define DE_CR 0.9f\n"
"__kernel void mutate_and_crossover(\n"
"    __global float* population,\n"
"    __global float* trial,\n"
"    __global const float* best,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed,\n"
"    __local float* local_best,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        // Cache best and bounds in local memory\n"
"        if (local_id < dim) {\n"
"            local_best[local_id] = best[local_id];\n"
"            local_bounds[2 * local_id] = bounds[2 * local_id];\n"
"            local_bounds[2 * local_id + 1] = bounds[2 * local_id + 1];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        // Select random indices\n"
"        int r1, r2, r3;\n"
"        do { r1 = lcg_rand_int(&local_seed, population_size); } while (r1 == id);\n"
"        do { r2 = lcg_rand_int(&local_seed, population_size); } while (r2 == id || r2 == r1);\n"
"        do { r3 = lcg_rand_int(&local_seed, population_size); } while (r3 == id || r3 == r1 || r3 == r2);\n"
"        int j_rand = lcg_rand_int(&local_seed, dim);\n"
"        // Mutate and crossover\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float v;\n"
"            switch (" DE_STRATEGY_STR ") {\n"
"                case 0: // STRATEGY_RAND_1_BIN\n"
"                    v = population[r1 * dim + d] + DE_F * (population[r2 * dim + d] - population[r3 * dim + d]);\n"
"                    break;\n"
"                case 1: // STRATEGY_BEST_1_BIN\n"
"                    v = local_best[d] + DE_F * (population[r1 * dim + d] - population[r2 * dim + d]);\n"
"                    break;\n"
"                case 2: // STRATEGY_RAND_TO_BEST_1\n"
"                    v = population[r1 * dim + d] + 0.5f * (local_best[d] - population[r1 * dim + d]) +\n"
"                        DE_F * (population[r2 * dim + d] - population[r3 * dim + d]);\n"
"                    break;\n"
"                default:\n"
"                    v = population[id * dim + d];\n"
"            }\n"
"            trial[id * dim + d] = (lcg_rand_float(&local_seed) < DE_CR || d == j_rand) ? clamp(v, local_bounds[2 * d], local_bounds[2 * d + 1]) : population[id * dim + d];\n"
"        }\n"
"    }\n"
"}\n"
"// Find best individual\n"
"__kernel void find_best(\n"
"    __global const float* fitness,\n"
"    __global float* best,\n"
"    __global float* best_fitness,\n"
"    __global const float* population,\n"
"    const int dim,\n"
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
"        int idx = local_indices[0];\n"
"        if (idx >= 0) {\n"
"            atomic_min((__global int*)best_fitness, as_int(local_fitness[0]));\n"
"            if (*best_fitness == local_fitness[0]) {\n"
"                for (int d = 0; d < dim; d++) {\n"
"                    best[d] = population[idx * dim + d];\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"}\n";

// Helper function to release OpenCL resources
static void release_cl_resources(
    cl_program program, cl_kernel init_kernel, cl_kernel mutate_kernel, cl_kernel best_kernel,
    cl_mem bounds_buffer, cl_mem trial_buffer, cl_mem best_buffer, cl_mem best_fitness_buffer)
{
    if (program) clReleaseProgram(program);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (mutate_kernel) clReleaseKernel(mutate_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (trial_buffer) clReleaseMemObject(trial_buffer);
    if (best_buffer) clReleaseMemObject(best_buffer);
    if (best_fitness_buffer) clReleaseMemObject(best_fitness_buffer);
}

// Main Optimization Function
void DE_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, mutate_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, trial_buffer = NULL, best_buffer = NULL, best_fitness_buffer = NULL;
    float *bounds_float = NULL, *trial = NULL, *best = NULL, *fitness = NULL, *best_fitness = NULL;
    double *temp = NULL;

    // Validate input
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        goto cleanup;
    }

    const int dim = opt->dim;
    const int population_size = opt->population_size;
    const int max_iter = opt->max_iter;
    if (dim < 1 || population_size < 1 || max_iter < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), or max_iter (%d)\n", 
                dim, population_size, max_iter);
        goto cleanup;
    }
    if (dim > 32) {
        fprintf(stderr, "Error: dim (%d) exceeds maximum supported value of 32\n", dim);
        goto cleanup;
    }

    // Query device capabilities
    size_t max_work_group_size;
    err = clGetDeviceInfo(opt->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying max work group size: %d\n", err);
        goto cleanup;
    }
    size_t local_work_size = max_work_group_size < 64 ? max_work_group_size : 64;

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = 0.0;
    }

    // Allocate host buffers
    bounds_float = (float *)malloc(2 * dim * sizeof(float));
    trial = (float *)malloc(population_size * dim * sizeof(float));
    best = (float *)malloc(dim * sizeof(float));
    fitness = (float *)malloc(population_size * sizeof(float));
    best_fitness = (float *)malloc(sizeof(float));
    temp = (double *)malloc(dim * sizeof(double));
    if (!bounds_float || !trial || !best || !fitness || !best_fitness || !temp) {
        fprintf(stderr, "Error allocating host buffers\n");
        goto cleanup;
    }

    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }
    *best_fitness = INFINITY;
    for (int i = 0; i < dim; i++) {
        best[i] = 0.0f;
    }

    // Create OpenCL program
    program = clCreateProgramWithSource(opt->context, 1, &de_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating DE program: %d\n", err);
        goto cleanup;
    }

    err = clBuildProgram(program, 1, &opt->device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building DE program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        goto cleanup;
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating init_kernel: %d\n", err); goto cleanup; }
    mutate_kernel = clCreateKernel(program, "mutate_and_crossover", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating mutate_kernel: %d\n", err); goto cleanup; }
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating best_kernel: %d\n", err); goto cleanup; }

    // Create buffers
    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                   2 * dim * sizeof(float), bounds_float, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating bounds_buffer: %d\n", err); goto cleanup; }
    trial_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, 
                                  population_size * dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating trial_buffer: %d\n", err); goto cleanup; }
    best_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                 dim * sizeof(float), best, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating best_buffer: %d\n", err); goto cleanup; }
    best_fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                        sizeof(float), best_fitness, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating best_fitness_buffer: %d\n", err); goto cleanup; }

    // Initialize population
    uint seed = (uint)time(NULL);
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 3, sizeof(int), &population_size);
    err |= clSetKernelArg(init_kernel, 4, sizeof(uint), &seed);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error setting init_kernel args: %d\n", err); goto cleanup; }

    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(opt->queue, init_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error enqueuing init_kernel: %d\n", err); goto cleanup; }

    // Evaluate initial fitness on host
    err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                              population_size * dim * sizeof(float), trial, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error reading population_buffer: %d\n", err); goto cleanup; }
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            temp[d] = (double)trial[i * dim + d];
        }
        fitness[i] = (float)objective_function(temp);
        if (fitness[i] < *best_fitness) {
            *best_fitness = fitness[i];
            for (int d = 0; d < dim; d++) {
                best[d] = trial[i * dim + d];
            }
        }
    }
    err = clEnqueueWriteBuffer(opt->queue, opt->fitness_buffer, CL_TRUE, 0, 
                               population_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error writing fitness_buffer: %d\n", err); goto cleanup; }
    err = clEnqueueWriteBuffer(opt->queue, best_buffer, CL_TRUE, 0, 
                               dim * sizeof(float), best, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error writing best_buffer: %d\n", err); goto cleanup; }
    err = clEnqueueWriteBuffer(opt->queue, best_fitness_buffer, CL_TRUE, 0, 
                               sizeof(float), best_fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error writing best_fitness_buffer: %d\n", err); goto cleanup; }

    // Main optimization loop
    for (int iter = 0; iter < max_iter && opt->best_solution.fitness > DE_TOL; iter++) {
        // Mutate and crossover
        seed += population_size;
        err = clSetKernelArg(mutate_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(mutate_kernel, 1, sizeof(cl_mem), &trial_buffer);
        err |= clSetKernelArg(mutate_kernel, 2, sizeof(cl_mem), &best_buffer);
        err |= clSetKernelArg(mutate_kernel, 3, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(mutate_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(mutate_kernel, 5, sizeof(int), &population_size);
        err |= clSetKernelArg(mutate_kernel, 6, sizeof(uint), &seed);
        err |= clSetKernelArg(mutate_kernel, 7, dim * sizeof(float), NULL);
        err |= clSetKernelArg(mutate_kernel, 8, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error setting mutate_kernel args: %d\n", err); goto cleanup; }

        err = clEnqueueNDRangeKernel(opt->queue, mutate_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error enqueuing mutate_kernel: %d\n", err); goto cleanup; }

        // Evaluate trial fitness on host
        err = clEnqueueReadBuffer(opt->queue, trial_buffer, CL_TRUE, 0, 
                                  population_size * dim * sizeof(float), trial, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error reading trial_buffer: %d\n", err); goto cleanup; }
        for (int i = 0; i < population_size; i++) {
            for (int d = 0; d < dim; d++) {
                temp[d] = (double)trial[i * dim + d];
            }
            float trial_fit = (float)objective_function(temp);
            if (trial_fit < fitness[i]) {
                for (int d = 0; d < dim; d++) {
                    trial[i * dim + d] = trial[i * dim + d];
                }
                fitness[i] = trial_fit;
            }
        }

        // Update population and fitness
        err = clEnqueueWriteBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                                   population_size * dim * sizeof(float), trial, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error writing population_buffer: %d\n", err); goto cleanup; }
        err = clEnqueueWriteBuffer(opt->queue, opt->fitness_buffer, CL_TRUE, 0, 
                                   population_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error writing fitness_buffer: %d\n", err); goto cleanup; }

        // Find best individual
        err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &best_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(cl_mem), &best_fitness_buffer);
        err |= clSetKernelArg(best_kernel, 3, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(best_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(best_kernel, 5, sizeof(int), &population_size);
        err |= clSetKernelArg(best_kernel, 6, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 7, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error setting best_kernel args: %d\n", err); goto cleanup; }

        err = clEnqueueNDRangeKernel(opt->queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error enqueuing best_kernel: %d\n", err); goto cleanup; }

        // Update best solution
        err = clEnqueueReadBuffer(opt->queue, best_fitness_buffer, CL_TRUE, 0, sizeof(float), best_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error reading best_fitness_buffer: %d\n", err); goto cleanup; }
        if (*best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)*best_fitness;
            err = clEnqueueReadBuffer(opt->queue, best_buffer, CL_TRUE, 0, dim * sizeof(float), best, 0, NULL, NULL);
            if (err != CL_SUCCESS) { fprintf(stderr, "Error reading best_buffer: %d\n", err); goto cleanup; }
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)best[d];
            }
        }

        printf("DE|%5d -----> %9.16f\n", iter + 1, opt->best_solution.fitness);
    }

    // Final best solution
    err = clEnqueueReadBuffer(opt->queue, best_fitness_buffer, CL_TRUE, 0, sizeof(float), best_fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error reading final best_fitness_buffer: %d\n", err); goto cleanup; }
    if (*best_fitness < opt->best_solution.fitness) {
        opt->best_solution.fitness = (double)*best_fitness;
        err = clEnqueueReadBuffer(opt->queue, best_buffer, CL_TRUE, 0, dim * sizeof(float), best, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error reading final best_buffer: %d\n", err); goto cleanup; }
        for (int d = 0; d < dim; d++) {
            opt->best_solution.position[d] = (double)best[d];
        }
    }

cleanup:
    free(bounds_float); bounds_float = NULL;
    free(trial); trial = NULL;
    free(best); best = NULL;
    free(fitness); fitness = NULL;
    free(best_fitness); best_fitness = NULL;
    free(temp); temp = NULL;
    release_cl_resources(program, init_kernel, mutate_kernel, best_kernel,
                         bounds_buffer, trial_buffer, best_buffer, best_fitness_buffer);

    clFinish(opt->queue);
    fprintf(stderr, "Cleanup completed\n");
}
