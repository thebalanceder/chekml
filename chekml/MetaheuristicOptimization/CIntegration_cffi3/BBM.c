/* BBM.c - GPU-Optimized Bee-Based Metaheuristic */
#include "BBM.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// OpenCL kernel for BBM
static const char* bbm_kernel_source =
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
"        int bee_idx = id / dim;\n"
"        int d = id % dim;\n"
"        uint local_seed = seed + id;\n"
"        float min = bounds[2 * d];\n"
"        float max = bounds[2 * d + 1];\n"
"        population[bee_idx * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"    }\n"
"}\n"
"// Mating and worker phases\n"
"#define BBM_CFR_FACTOR 9.435f // Matches BBM.h\n"
"__kernel void mating_worker_phase(\n"
"    __global float* population,\n"
"    __global const float* queen,\n"
"    __global float* temp_solutions,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed,\n"
"    __local float* local_queen)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id;\n"
"        // Cache queen in local memory\n"
"        if (local_id == 0) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                local_queen[d] = queen[d];\n"
"            }\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        // Select random drone\n"
"        int drone_idx = lcg_rand(&local_seed) % population_size;\n"
"        // Mating phase: Blend-alpha crossover\n"
"        float temp[32]; // Assumes dim <= 32\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float q = local_queen[d];\n"
"            float dr = population[drone_idx * dim + d];\n"
"            float diff = fabs(q - dr);\n"
"            float lower = fmin(q, dr) - 0.5f * diff;\n"
"            float upper = fmax(q, dr) + 0.5f * diff;\n"
"            temp[d] = lower + lcg_rand_float(&local_seed, 0.0f, 1.0f) * (upper - lower);\n"
"        }\n"
"        // Mating velocity adjustment\n"
"        float r1 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float r2 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float Vi = (r1 < 0.2f) ? \n"
"                   (pow(0.3f, 2.0f / 3.0f) / 1.2f * r1) : \n"
"                   (pow(0.3f, 2.0f / 3.0f) / 1.2f * r2);\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float new_pos = local_queen[d] + (temp[d] - local_queen[d]) * Vi * lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"            temp_solutions[id * dim + d] = clamp(new_pos, bounds[2 * d], bounds[2 * d + 1]);\n"
"        }\n"
"        // Worker phase\n"
"        float r3 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float r4 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float CFR = BBM_CFR_FACTOR * lcg_rand_float(&local_seed, 0.0f, 1.0f) * 2.5f;\n"
"        float Vi2 = (r3 < 0.46f) ? \n"
"                    (pow(1.35f, 2.0f / 3.0f) / (2.0f * CFR) * r3) : \n"
"                    (pow(1.35f, 2.0f / 3.0f) / (2.0f * CFR) * r4);\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float pos = temp_solutions[id * dim + d];\n"
"            float improve = (lcg_rand_float(&local_seed, 0.0f, 1.0f) * (local_queen[d] - pos));\n"
"            float new_pos = local_queen[d] + (local_queen[d] - pos) * Vi2 + improve;\n"
"            population[id * dim + d] = clamp(new_pos, bounds[2 * d], bounds[2 * d + 1]);\n"
"        }\n"
"    }\n"
"}\n"
"// Replacement phase\n"
"__kernel void replacement_phase(\n"
"    __global float* population,\n"
"    __global const float* bounds,\n"
"    __global const int* worst_indices,\n"
"    const int dim,\n"
"    const int worst_count,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < worst_count) {\n"
"        int bee_idx = worst_indices[id];\n"
"        uint local_seed = seed + id;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            population[bee_idx * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"        }\n"
"    }\n"
"}\n";

void BBM_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, mating_worker_kernel = NULL, replacement_kernel = NULL;
    cl_mem bounds_buffer = NULL, queen_buffer = NULL, temp_buffer = NULL, worst_indices_buffer = NULL;
    float* bounds_float = NULL, *population = NULL, *fitness = NULL, *queen_float = NULL;
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
    const int worst_count = (int)(0.23f * population_size);
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
    program = clCreateProgramWithSource(opt->context, 1, &bbm_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating BBM program: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building BBM program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    mating_worker_kernel = clCreateKernel(program, "mating_worker_phase", &err);
    replacement_kernel = clCreateKernel(program, "replacement_phase", &err);
    if (err != CL_SUCCESS || !init_kernel || !mating_worker_kernel || !replacement_kernel) {
        fprintf(stderr, "Error creating BBM kernels: %d\n", err);
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
    queen_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    temp_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    worst_indices_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, worst_count * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !queen_buffer || !temp_buffer || !worst_indices_buffer) {
        fprintf(stderr, "Error creating BBM buffers: %d\n", err);
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
    worst_indices = (int*)malloc(worst_count * sizeof(int));
    queen_float = (float*)malloc(dim * sizeof(float));
    if (!population || !fitness || !worst_indices || !queen_float) {
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

        // Evaluate fitness
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

        // Write fitness
        err = clEnqueueWriteBuffer(opt->queue, opt->fitness_buffer, CL_TRUE, 0, 
                                   population_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            goto cleanup;
        }

        // Queen selection (host-side reduction)
        int min_idx = 0;
        float min_fitness = fitness[0];
        for (int i = 1; i < population_size; i++) {
            if (fitness[i] < min_fitness) {
                min_fitness = fitness[i];
                min_idx = i;
            }
        }
        if (min_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)min_fitness;
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)population[min_idx * dim + d];
                queen_float[d] = (float)opt->best_solution.position[d];
            }
        }

        // Write queen to GPU
        err = clEnqueueWriteBuffer(opt->queue, queen_buffer, CL_TRUE, 0, 
                                   dim * sizeof(float), queen_float, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing queen buffer: %d\n", err);
            goto cleanup;
        }

        // Mating and worker phase
        err = clSetKernelArg(mating_worker_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(mating_worker_kernel, 1, sizeof(cl_mem), &queen_buffer);
        err |= clSetKernelArg(mating_worker_kernel, 2, sizeof(cl_mem), &temp_buffer);
        err |= clSetKernelArg(mating_worker_kernel, 3, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(mating_worker_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(mating_worker_kernel, 5, sizeof(int), &population_size);
        err |= clSetKernelArg(mating_worker_kernel, 6, sizeof(uint), &seed);
        err |= clSetKernelArg(mating_worker_kernel, 7, dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting mating_worker kernel args: %d\n", err);
            goto cleanup;
        }

        size_t mating_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, mating_worker_kernel, 1, NULL, &mating_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing mating_worker kernel: %d\n", err);
            goto cleanup;
        }

        // Find worst indices (host-side for simplicity)
        for (int i = 0; i < population_size; i++) {
            fitness[i] = -fitness[i]; // Max-heap for worst
        }
        for (int i = 0; i < worst_count; i++) {
            int max_idx = i;
            for (int j = i + 1; j < population_size; j++) {
                if (fitness[j] > fitness[max_idx]) {
                    max_idx = j;
                }
            }
            worst_indices[i] = max_idx;
            float temp = fitness[i];
            fitness[i] = fitness[max_idx];
            fitness[max_idx] = temp;
        }
        for (int i = 0; i < population_size; i++) {
            fitness[i] = -fitness[i]; // Restore
        }

        // Write worst indices
        err = clEnqueueWriteBuffer(opt->queue, worst_indices_buffer, CL_TRUE, 0, 
                                   worst_count * sizeof(int), worst_indices, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing worst indices buffer: %d\n", err);
            goto cleanup;
        }

        // Replacement phase
        err = clSetKernelArg(replacement_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(replacement_kernel, 1, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(replacement_kernel, 2, sizeof(cl_mem), &worst_indices_buffer);
        err |= clSetKernelArg(replacement_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(replacement_kernel, 4, sizeof(int), &worst_count);
        err |= clSetKernelArg(replacement_kernel, 5, sizeof(uint), &seed);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting replacement kernel args: %d\n", err);
            goto cleanup;
        }

        size_t replacement_global_work_size = ((worst_count + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, replacement_kernel, 1, NULL, &replacement_global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing replacement kernel: %d\n", err);
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
    if (worst_indices) free(worst_indices);
    if (queen_float) free(queen_float);
    if (bounds_float) free(bounds_float);
    population = NULL;
    fitness = NULL;
    worst_indices = NULL;
    queen_float = NULL;
    bounds_float = NULL;

    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (queen_buffer) clReleaseMemObject(queen_buffer);
    if (temp_buffer) clReleaseMemObject(temp_buffer);
    if (worst_indices_buffer) clReleaseMemObject(worst_indices_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (mating_worker_kernel) clReleaseKernel(mating_worker_kernel);
    if (replacement_kernel) clReleaseKernel(replacement_kernel);
    if (program) clReleaseProgram(program);
    clFinish(opt->queue);
}
