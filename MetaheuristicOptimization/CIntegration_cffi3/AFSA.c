/* AFSA.c - GPU-Optimized Artificial Fish Swarm Algorithm */
#include "AFSA.h"
#include <string.h>
#include <float.h>
#include <omp.h>

// OpenCL kernel for AFSA
static const char* afsa_kernel_source = 
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
"    if (id < population_size) {\n"
"        uint local_seed = seed + id;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            population[id * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"        }\n"
"    }\n"
"}\n"
"// Prey behavior candidates\n"
"__kernel void prey_behavior(\n"
"    __global float* population,\n"
"    __global float* candidates,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size * 5) {\n"
"        int fish_idx = id / 5;\n"
"        int try_idx = id % 5;\n"
"        uint local_seed = seed + id;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float xi = population[fish_idx * dim + d];\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            float candidate = xi + 0.3f * lcg_rand_float(&local_seed, -1.0f, 1.0f);\n"
"            candidates[id * dim + d] = clamp(candidate, min, max);\n"
"        }\n"
"    }\n"
"}\n"
"// Compute neighbor centers for swarm behavior\n"
"__kernel void swarm_behavior(\n"
"    __global float* population,\n"
"    __global float* centers,\n"
"    __global int* neighbor_counts,\n"
"    const int dim,\n"
"    const int population_size)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        float center[32]; // Assumes dim <= 32\n"
"        for (int d = 0; d < dim; d++) center[d] = 0.0f;\n"
"        int count = 0;\n"
"        for (int j = 0; j < population_size; j++) {\n"
"            if (j == id) continue;\n"
"            float dist = 0.0f;\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float diff = population[j * dim + d] - population[id * dim + d];\n"
"                dist += diff * diff;\n"
"            }\n"
"            if (dist < 0.3f * 0.3f) {\n"
"                for (int d = 0; d < dim; d++) {\n"
"                    center[d] += population[j * dim + d];\n"
"                }\n"
"                count++;\n"
"            }\n"
"        }\n"
"        neighbor_counts[id] = count;\n"
"        if (count > 0) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                centers[id * dim + d] = center[d] / count;\n"
"            }\n"
"        } else {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                centers[id * dim + d] = population[id * dim + d];\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"// Find best neighbor for follow behavior\n"
"__kernel void follow_behavior(\n"
"    __global float* population,\n"
"    __global float* fitness,\n"
"    __global float* best_neighbors,\n"
"    const int dim,\n"
"    const int population_size)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        float best_fitness = fitness[id];\n"
"        bool found = false;\n"
"        for (int j = 0; j < population_size; j++) {\n"
"            if (j == id) continue;\n"
"            float dist = 0.0f;\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float diff = population[j * dim + d] - population[id * dim + d];\n"
"                dist += diff * diff;\n"
"            }\n"
"            if (dist < 0.3f * 0.3f) {\n"
"                if (fitness[j] < best_fitness) {\n"
"                    best_fitness = fitness[j];\n"
"                    for (int d = 0; d < dim; d++) {\n"
"                        best_neighbors[id * dim + d] = population[j * dim + d];\n"
"                    }\n"
"                    found = true;\n"
"                }\n"
"            }\n"
"        }\n"
"        if (!found) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                best_neighbors[id * dim + d] = population[id * dim + d];\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"// Update positions based on behavior\n"
"__kernel void update_positions(\n"
"    __global float* population,\n"
"    __global const float* targets,\n"
"    __global const float* bounds,\n"
"    __global const int* behavior_choices,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id;\n"
"        int behavior = behavior_choices[id];\n"
"        if (behavior == 0) return; // Prey handled separately\n"
"        float norm = 0.0f;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float diff = targets[id * dim + d] - population[id * dim + d];\n"
"            norm += diff * diff;\n"
"        }\n"
"        norm = sqrt(norm);\n"
"        if (norm == 0.0f) return;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float step = 0.1f * (targets[id * dim + d] - population[id * dim + d]) / norm * lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"            float new_pos = population[id * dim + d] + step;\n"
"            population[id * dim + d] = clamp(new_pos, bounds[2 * d], bounds[2 * d + 1]);\n"
"        }\n"
"    }\n"
"}\n";

void AFSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, prey_kernel = NULL, swarm_kernel = NULL, follow_kernel = NULL, update_kernel = NULL;
    cl_mem candidates_buffer = NULL, centers_buffer = NULL, neighbors_buffer = NULL, counts_buffer = NULL, bounds_buffer = NULL, choices_buffer = NULL;

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
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), or max_iter (%d)\n", dim, population_size, max_iter);
        exit(EXIT_FAILURE);
    }

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = 0.0;
    }

    // Create OpenCL program
    program = clCreateProgramWithSource(opt->context, 1, &afsa_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating AFSA program: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building AFSA program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    prey_kernel = clCreateKernel(program, "prey_behavior", &err);
    swarm_kernel = clCreateKernel(program, "swarm_behavior", &err);
    follow_kernel = clCreateKernel(program, "follow_behavior", &err);
    update_kernel = clCreateKernel(program, "update_positions", &err);
    if (err != CL_SUCCESS || !init_kernel || !prey_kernel || !swarm_kernel || !follow_kernel || !update_kernel) {
        fprintf(stderr, "Error creating AFSA kernels: %d\n", err);
        clReleaseKernel(init_kernel);
        clReleaseKernel(prey_kernel);
        clReleaseKernel(swarm_kernel);
        clReleaseKernel(follow_kernel);
        clReleaseKernel(update_kernel);
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    // Create buffers
    float* bounds_float = (float*)malloc(2 * dim * sizeof(float));
    if (!bounds_float) {
        fprintf(stderr, "Error: Memory allocation failed for bounds_float\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }

    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    candidates_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * TRY_NUMBER * dim * sizeof(float), NULL, &err);
    centers_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    neighbors_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    counts_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    choices_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, population_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !candidates_buffer || !centers_buffer || !neighbors_buffer || !counts_buffer || !choices_buffer) {
        fprintf(stderr, "Error creating AFSA buffers: %d\n", err);
        free(bounds_float);
        goto cleanup;
    }

    // Write bounds
    err = clEnqueueWriteBuffer(opt->queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bounds buffer: %d\n", err);
        free(bounds_float);
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
        free(bounds_float);
        goto cleanup;
    }

    size_t global_work_size = population_size;
    err = clEnqueueNDRangeKernel(opt->queue, init_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel: %d\n", err);
        free(bounds_float);
        goto cleanup;
    }

    // Main loop
    float* population = (float*)malloc(population_size * dim * sizeof(float));
    float* candidates = (float*)malloc(population_size * TRY_NUMBER * dim * sizeof(float));
    float* centers = (float*)malloc(population_size * dim * sizeof(float));
    float* neighbors = (float*)malloc(population_size * dim * sizeof(float));
    float* fitness = (float*)malloc(population_size * sizeof(float));
    float* candidate_fitness = (float*)malloc(population_size * TRY_NUMBER * sizeof(float));
    float* center_fitness = (float*)malloc(population_size * sizeof(float));
    int* neighbor_counts = (int*)malloc(population_size * sizeof(int));
    int* behavior_choices = (int*)malloc(population_size * sizeof(int));
    if (!population || !candidates || !centers || !neighbors || !fitness || !candidate_fitness || !center_fitness || !neighbor_counts || !behavior_choices) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(population);
        free(candidates);
        free(centers);
        free(neighbors);
        free(fitness);
        free(candidate_fitness);
        free(center_fitness);
        free(neighbor_counts);
        free(behavior_choices);
        free(bounds_float);
        goto cleanup;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        // Read current population
        err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                                  population_size * dim * sizeof(float), population, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading population buffer: %d\n", err);
            goto cleanup_loop;
        }

        // Evaluate current population
        #pragma omp parallel for
        for (int i = 0; i < population_size; i++) {
            double* pos_double = (double*)malloc(dim * sizeof(double));
            for (int d = 0; d < dim; d++) {
                pos_double[d] = (double)population[i * dim + d];
            }
            fitness[i] = (float)objective_function(pos_double);
            free(pos_double);
        }

        // Update best solution
        for (int i = 0; i < population_size; i++) {
            if (fitness[i] < opt->best_solution.fitness) {
                opt->best_solution.fitness = (double)fitness[i];
                for (int d = 0; d < dim; d++) {
                    opt->best_solution.position[d] = (double)population[i * dim + d];
                }
            }
        }

        // Write fitness to GPU
        err = clEnqueueWriteBuffer(opt->queue, opt->fitness_buffer, CL_TRUE, 0, 
                                   population_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            goto cleanup_loop;
        }

        // Generate behavior choices
        for (int i = 0; i < population_size; i++) {
            behavior_choices[i] = rand() % 3; // 0: prey, 1: swarm, 2: follow
        }
        err = clEnqueueWriteBuffer(opt->queue, choices_buffer, CL_TRUE, 0, 
                                   population_size * sizeof(int), behavior_choices, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing choices buffer: %d\n", err);
            goto cleanup_loop;
        }

        // Prey behavior
        err = clSetKernelArg(prey_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(prey_kernel, 1, sizeof(cl_mem), &candidates_buffer);
        err |= clSetKernelArg(prey_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(prey_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(prey_kernel, 4, sizeof(int), &population_size);
        err |= clSetKernelArg(prey_kernel, 5, sizeof(uint), &seed);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting prey kernel args: %d\n", err);
            goto cleanup_loop;
        }

        global_work_size = population_size * TRY_NUMBER;
        err = clEnqueueNDRangeKernel(opt->queue, prey_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing prey kernel: %d\n", err);
            goto cleanup_loop;
        }

        // Read candidates
        err = clEnqueueReadBuffer(opt->queue, candidates_buffer, CL_TRUE, 0, 
                                  population_size * TRY_NUMBER * dim * sizeof(float), candidates, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading candidates buffer: %d\n", err);
            goto cleanup_loop;
        }

        // Evaluate candidates
        #pragma omp parallel for
        for (int i = 0; i < population_size * TRY_NUMBER; i++) {
            double* pos_double = (double*)malloc(dim * sizeof(double));
            for (int d = 0; d < dim; d++) {
                pos_double[d] = (double)candidates[i * dim + d];
            }
            candidate_fitness[i] = (float)objective_function(pos_double);
            free(pos_double);
        }

        // Select best candidate for prey behavior
        float* best_candidates = (float*)malloc(population_size * dim * sizeof(float));
        if (!best_candidates) {
            fprintf(stderr, "Error: Memory allocation failed for best_candidates\n");
            goto cleanup_loop;
        }
        for (int i = 0; i < population_size; i++) {
            float best_fitness = fitness[i];
            int best_idx = i * TRY_NUMBER;
            for (int t = 0; t < TRY_NUMBER; t++) {
                int idx = i * TRY_NUMBER + t;
                if (candidate_fitness[idx] < best_fitness) {
                    best_fitness = candidate_fitness[idx];
                    best_idx = idx;
                }
            }
            for (int d = 0; d < dim; d++) {
                best_candidates[i * dim + d] = candidates[best_idx * dim + d];
            }
        }

        // Swarm behavior
        err = clSetKernelArg(swarm_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(swarm_kernel, 1, sizeof(cl_mem), &centers_buffer);
        err |= clSetKernelArg(swarm_kernel, 2, sizeof(cl_mem), &counts_buffer);
        err |= clSetKernelArg(swarm_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(swarm_kernel, 4, sizeof(int), &population_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting swarm kernel args: %d\n", err);
            free(best_candidates);
            goto cleanup_loop;
        }

        global_work_size = population_size;
        err = clEnqueueNDRangeKernel(opt->queue, swarm_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing swarm kernel: %d\n", err);
            free(best_candidates);
            goto cleanup_loop;
        }

        // Read centers and counts
        err = clEnqueueReadBuffer(opt->queue, centers_buffer, CL_TRUE, 0, 
                                  population_size * dim * sizeof(float), centers, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(opt->queue, counts_buffer, CL_TRUE, 0, 
                                   population_size * sizeof(int), neighbor_counts, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading centers/counts buffers: %d\n", err);
            free(best_candidates);
            goto cleanup_loop;
        }

        // Evaluate centers
        #pragma omp parallel for
        for (int i = 0; i < population_size; i++) {
            if (neighbor_counts[i] == 0) {
                center_fitness[i] = fitness[i];
                continue;
            }
            double* pos_double = (double*)malloc(dim * sizeof(double));
            for (int d = 0; d < dim; d++) {
                pos_double[d] = (double)centers[i * dim + d];
            }
            center_fitness[i] = (float)objective_function(pos_double);
            free(pos_double);
        }

        // Follow behavior
        err = clSetKernelArg(follow_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(follow_kernel, 1, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(follow_kernel, 2, sizeof(cl_mem), &neighbors_buffer);
        err |= clSetKernelArg(follow_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(follow_kernel, 4, sizeof(int), &population_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting follow kernel args: %d\n", err);
            free(best_candidates);
            goto cleanup_loop;
        }

        global_work_size = population_size;
        err = clEnqueueNDRangeKernel(opt->queue, follow_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing follow kernel: %d\n", err);
            free(best_candidates);
            goto cleanup_loop;
        }

        // Read neighbors
        err = clEnqueueReadBuffer(opt->queue, neighbors_buffer, CL_TRUE, 0, 
                                  population_size * dim * sizeof(float), neighbors, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading neighbors buffer: %d\n", err);
            free(best_candidates);
            goto cleanup_loop;
        }

        // Decide target positions
        float* targets = (float*)malloc(population_size * dim * sizeof(float));
        if (!targets) {
            fprintf(stderr, "Error: Memory allocation failed for targets\n");
            free(best_candidates);
            goto cleanup_loop;
        }
        for (int i = 0; i < population_size; i++) {
            int behavior = behavior_choices[i];
            if (behavior == 0) { // Prey
                if (candidate_fitness[i * TRY_NUMBER] < fitness[i]) {
                    for (int d = 0; d < dim; d++) {
                        targets[i * dim + d] = best_candidates[i * dim + d];
                    }
                } else {
                    for (int d = 0; d < dim; d++) {
                        targets[i * dim + d] = population[i * dim + d];
                    }
                }
            } else if (behavior == 1) { // Swarm
                if (neighbor_counts[i] > 0 && center_fitness[i] / neighbor_counts[i] < fitness[i] * DELTA) {
                    for (int d = 0; d < dim; d++) {
                        targets[i * dim + d] = centers[i * dim + d];
                    }
                } else {
                    for (int d = 0; d < dim; d++) {
                        targets[i * dim + d] = best_candidates[i * dim + d];
                    }
                }
            } else { // Follow
                float neighbor_fitness = 0.0f;
                double* pos_double = (double*)malloc(dim * sizeof(double));
                for (int d = 0; d < dim; d++) {
                    pos_double[d] = (double)neighbors[i * dim + d];
                }
                neighbor_fitness = (float)objective_function(pos_double);
                free(pos_double);
                if (neighbor_fitness < fitness[i] * DELTA) {
                    for (int d = 0; d < dim; d++) {
                        targets[i * dim + d] = neighbors[i * dim + d];
                    }
                } else {
                    for (int d = 0; d < dim; d++) {
                        targets[i * dim + d] = best_candidates[i * dim + d];
                    }
                }
            }
        }

        // Write targets for update
        err = clEnqueueWriteBuffer(opt->queue, candidates_buffer, CL_TRUE, 0, 
                                   population_size * dim * sizeof(float), targets, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing targets buffer: %d\n", err);
            free(best_candidates);
            free(targets);
            goto cleanup_loop;
        }

        // Update positions
        err = clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &candidates_buffer);
        err |= clSetKernelArg(update_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(update_kernel, 3, sizeof(cl_mem), &choices_buffer);
        err |= clSetKernelArg(update_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(update_kernel, 5, sizeof(int), &population_size);
        err |= clSetKernelArg(update_kernel, 6, sizeof(uint), &seed);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting update kernel args: %d\n", err);
            free(best_candidates);
            free(targets);
            goto cleanup_loop;
        }

        global_work_size = population_size;
        err = clEnqueueNDRangeKernel(opt->queue, update_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing update kernel: %d\n", err);
            free(best_candidates);
            free(targets);
            goto cleanup_loop;
        }

        free(best_candidates);
        free(targets);
    }

cleanup_loop:
    free(population);
    free(candidates);
    free(centers);
    free(neighbors);
    free(fitness);
    free(candidate_fitness);
    free(center_fitness);
    free(neighbor_counts);
    free(behavior_choices);

cleanup:
    free(bounds_float);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (candidates_buffer) clReleaseMemObject(candidates_buffer);
    if (centers_buffer) clReleaseMemObject(centers_buffer);
    if (neighbors_buffer) clReleaseMemObject(neighbors_buffer);
    if (counts_buffer) clReleaseMemObject(counts_buffer);
    if (choices_buffer) clReleaseMemObject(choices_buffer);
    clReleaseKernel(init_kernel);
    clReleaseKernel(prey_kernel);
    clReleaseKernel(swarm_kernel);
    clReleaseKernel(follow_kernel);
    clReleaseKernel(update_kernel);
    clReleaseProgram(program);
    clFinish(opt->queue);
}
