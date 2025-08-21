#include "ALO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CL/cl.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

// Portable timing function
static double get_time_ms() {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / freq.QuadPart * 1000.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#endif
}

// OpenCL kernel source for ALO
static const char *alo_kernel_source =
"#define I_FACTOR_1 100.0f\n"
"#define I_FACTOR_2 1000.0f\n"
"#define I_FACTOR_3 10000.0f\n"
"#define I_FACTOR_4 100000.0f\n"
"#define I_FACTOR_5 1000000.0f\n"
"#define ROULETTE_EPSILON 1e-10f\n"
// XORShift random number generator
"uint xorshift(uint *rng_state) {\n"
"    uint x = *rng_state;\n"
"    x ^= x << 13;\n"
"    x ^= x >> 17;\n"
"    x ^= x << 5;\n"
"    *rng_state = x;\n"
"    return x;\n"
"}\n"
"float rand_float(uint *rng_state, float min, float max) {\n"
"    uint r = xorshift(rng_state);\n"
"    return min + (max - min) * ((float)r / 0x7FFFu);\n"
"}\n"
// Kernel to initialize populations
"__kernel void initialize_populations(\n"
"    __global float *positions,\n"
"    __global float *bounds,\n"
"    __global uint *rng_states,\n"
"    const int dim,\n"
"    const int pop_size)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= pop_size) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float lb = bounds[2 * d];\n"
"        float ub = bounds[2 * d + 1];\n"
"        positions[idx * dim + d] = rand_float(&rng_state, lb, ub);\n"
"    }\n"
"    rng_states[idx] = rng_state;\n"
"}\n"
// Kernel for roulette wheel selection weights
"__kernel void compute_roulette_weights(\n"
"    __global float *fitness,\n"
"    __global float *weights,\n"
"    __global float *cumsum,\n"
"    const int size)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= size) return;\n"
"    weights[idx] = 1.0f / (fitness[idx] + ROULETTE_EPSILON);\n"
"    cumsum[idx] = weights[idx];\n"
"    if (idx > 0) cumsum[idx] += cumsum[idx - 1];\n"
"}\n"
// Kernel for random walk around antlion
"__kernel void random_walk_phase(\n"
"    __global float *positions,\n"
"    __global float *antlion_pos,\n"
"    __global float *bounds,\n"
"    __global uint *rng_states,\n"
"    __global float *walks,\n"
"    const int t,\n"
"    const int max_iter,\n"
"    const int dim,\n"
"    const int pop_size)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= pop_size) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    float I = 1.0f;\n"
"    float T = (float)max_iter;\n"
"    if (t > T / 10.0f) I = 1.0f + I_FACTOR_1 * (t / T);\n"
"    if (t > T / 2.0f) I = 1.0f + I_FACTOR_2 * (t / T);\n"
"    if (t > T * 3.0f / 4.0f) I = 1.0f + I_FACTOR_3 * (t / T);\n"
"    if (t > T * 0.9f) I = 1.0f + I_FACTOR_4 * (t / T);\n"
"    if (t > T * 0.95f) I = 1.0f + I_FACTOR_5 * (t / T);\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float lb = bounds[2 * d] / I;\n"
"        float ub = bounds[2 * d + 1] / I;\n"
"        float r = rand_float(&rng_state, 0.0f, 1.0f);\n"
"        if (r < 0.5f) lb += antlion_pos[d];\n"
"        else lb = -lb + antlion_pos[d];\n"
"        r = rand_float(&rng_state, 0.0f, 1.0f);\n"
"        if (r >= 0.5f) ub += antlion_pos[d];\n"
"        else ub = -ub + antlion_pos[d];\n"
"        float X = 0.0f, a = 0.0f, b = 0.0f;\n"
"        for (int k = 0; k < max_iter; k++) {\n"
"            r = rand_float(&rng_state, 0.0f, 1.0f);\n"
"            X += (r > 0.5f ? 1.0f : -1.0f);\n"
"            if (X < a) a = X;\n"
"            if (X > b) b = X;\n"
"        }\n"
"        float lower = lb, upper = ub;\n"
"        walks[idx * dim + d] = ((X - a) * (upper - lower)) / (b - a + 1e-10f) + lower;\n"
"    }\n"
"    rng_states[idx] = rng_state;\n"
"}\n"
// Kernel to update ant positions
"__kernel void update_ant_positions(\n"
"    __global float *positions,\n"
"    __global float *antlion_pos,\n"
"    __global float *elite_pos,\n"
"    __global float *weights,\n"
"    __global float *cumsum,\n"
"    __global uint *rng_states,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    const int antlion_size)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= pop_size) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    float total = cumsum[antlion_size - 1];\n"
"    float p = rand_float(&rng_state, 0.0f, total);\n"
"    int roulette_idx = 0;\n"
"    for (int i = 0; i < antlion_size; i++) {\n"
"        if (cumsum[i] > p) {\n"
"            roulette_idx = i;\n"
"            break;\n"
"        }\n"
"    }\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float ra = antlion_pos[roulette_idx * dim + d];\n"
"        float re = elite_pos[d];\n"
"        positions[idx * dim + d] = (ra + re) / 2.0f;\n"
"    }\n"
"    rng_states[idx] = rng_state;\n"
"}\n"
// Kernel to update antlions (simplified sorting-based selection)
"__kernel void update_antlions_phase(\n"
"    __global float *antlion_pos,\n"
"    __global float *positions,\n"
"    __global float *fitness,\n"
"    __global float *elite_pos,\n"
"    __global float *elite_fitness,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    const int antlion_size)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= antlion_size) return;\n"
"    float min_fitness = elite_fitness[0];\n"
"    int min_idx = 0;\n"
"    for (int i = 0; i < pop_size; i++) {\n"
"        if (fitness[i] < min_fitness) {\n"
"            min_fitness = fitness[i];\n"
"            min_idx = i;\n"
"        }\n"
"    }\n"
"    for (int d = 0; d < dim; d++) {\n"
"        antlion_pos[idx * dim + d] = (idx == 0) ? elite_pos[d] : positions[min_idx * dim + d];\n"
"    }\n"
"}\n";

// Main Optimization Function
void ALO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;

    // Validate input
    if (!opt || !opt->population || !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: Population[%d].position is null\n", i);
            return;
        }
    }

    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    const int max_iter = opt->max_iter;
    if (dim < 1 || pop_size < 1 || max_iter < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), pop_size (%d), or max_iter (%d)\n", 
                dim, pop_size, max_iter);
        return;
    }

    // Initialize OpenCL program and kernels
    cl_program program = clCreateProgramWithSource(opt->context, 1, &alo_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating program: %d\n", err);
        return;
    }

    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        return;
    }

    cl_kernel init_kernel = clCreateKernel(program, "initialize_populations", &err);
    cl_kernel weights_kernel = clCreateKernel(program, "compute_roulette_weights", &err);
    cl_kernel walk_kernel = clCreateKernel(program, "random_walk_phase", &err);
    cl_kernel update_ants_kernel = clCreateKernel(program, "update_ant_positions", &err);
    cl_kernel update_antlions_kernel = clCreateKernel(program, "update_antlions_phase", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating kernels: %d\n", err);
        clReleaseProgram(program);
        return;
    }

    // Allocate GPU buffers
    cl_mem antlion_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, 
                                          pop_size * dim * sizeof(float), NULL, &err);
    cl_mem bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, 
                                         2 * dim * sizeof(float), NULL, &err);
    cl_mem rng_states_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, 
                                             pop_size * sizeof(uint), NULL, &err);
    cl_mem weights_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, 
                                          pop_size * sizeof(float), NULL, &err);
    cl_mem cumsum_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, 
                                         pop_size * sizeof(float), NULL, &err);
    cl_mem walks_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, 
                                        pop_size * dim * sizeof(float), NULL, &err);
    cl_mem elite_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, 
                                        dim * sizeof(float), NULL, &err);
    cl_mem elite_fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, 
                                                sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffers: %d\n", err);
        clReleaseKernel(init_kernel);
        clReleaseKernel(weights_kernel);
        clReleaseKernel(walk_kernel);
        clReleaseKernel(update_ants_kernel);
        clReleaseKernel(update_antlions_kernel);
        clReleaseProgram(program);
        return;
    }

    // Initialize host data
    float *bounds = (float *)malloc(2 * dim * sizeof(float));
    uint *rng_states = (uint *)malloc(pop_size * sizeof(uint));
    if (!bounds || !rng_states) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }

    for (int d = 0; d < dim; d++) {
        bounds[2 * d] = (float)opt->bounds[2 * d];
        bounds[2 * d + 1] = (float)opt->bounds[2 * d + 1];
    }
    for (int i = 0; i < pop_size; i++) {
        rng_states[i] = (uint)(i + 1) * 12345U;
    }

    // Write initial data to GPU
    err = clEnqueueWriteBuffer(opt->queue, bounds_buffer, CL_TRUE, 0, 
                              2 * dim * sizeof(float), bounds, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(opt->queue, rng_states_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(uint), rng_states, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing buffers: %d\n", err);
        goto cleanup;
    }

    // Initialize populations
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(cl_mem), &rng_states_buffer);
    err |= clSetKernelArg(init_kernel, 3, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 4, sizeof(int), &pop_size);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting init kernel args: %d\n", err);
        goto cleanup;
    }

    size_t global_work_size = pop_size;
    size_t local_work_size = 64;
    global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(opt->queue, init_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel: %d\n", err);
        goto cleanup;
    }

    // Copy population to antlion buffer
    err = clEnqueueCopyBuffer(opt->queue, opt->population_buffer, antlion_buffer, 0, 0, 
                             pop_size * dim * sizeof(float), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error copying population to antlion buffer: %d\n", err);
        goto cleanup;
    }

    // Profiling setup
    cl_event events[4] = {0}; // Events for weights, walk, update ants, update antlions
    double start_time, end_time;

    // Main optimization loop
    for (int t = 0; t < max_iter; t++) {
        start_time = get_time_ms();

        // Evaluate fitness on CPU
        float *positions = (float *)malloc(pop_size * dim * sizeof(float));
        if (!positions) {
            fprintf(stderr, "Error: Memory allocation failed for positions\n");
            goto cleanup;
        }
        err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                                 pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading population buffer: %d\n", err);
            free(positions);
            goto cleanup;
        }

        float *fitness = (float *)malloc(pop_size * sizeof(float));
        if (!fitness) {
            fprintf(stderr, "Error: Memory allocation failed for fitness\n");
            free(positions);
            goto cleanup;
        }
        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            if (!pos) {
                fprintf(stderr, "Error: Memory allocation failed for pos\n");
                free(fitness);
                free(positions);
                goto cleanup;
            }
            for (int d = 0; d < dim; d++) {
                pos[d] = (double)positions[i * dim + d];
            }
            opt->population[i].fitness = objective_function(pos);
            fitness[i] = (float)opt->population[i].fitness;
            free(pos);
        }
        free(positions);

        // Write fitness to GPU
        err = clEnqueueWriteBuffer(opt->queue, opt->fitness_buffer, CL_TRUE, 0, 
                                  pop_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            free(fitness);
            goto cleanup;
        }

        // Compute roulette weights
        err = clSetKernelArg(weights_kernel, 0, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(weights_kernel, 1, sizeof(cl_mem), &weights_buffer);
        err |= clSetKernelArg(weights_kernel, 2, sizeof(cl_mem), &cumsum_buffer);
        err |= clSetKernelArg(weights_kernel, 3, sizeof(int), &pop_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting weights kernel args: %d\n", err);
            free(fitness);
            goto cleanup;
        }

        global_work_size = pop_size;
        global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, weights_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[0]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing weights kernel: %d\n", err);
            free(fitness);
            goto cleanup;
        }

        // Random walk around antlions
        err = clSetKernelArg(walk_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(walk_kernel, 1, sizeof(cl_mem), &antlion_buffer);
        err |= clSetKernelArg(walk_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(walk_kernel, 3, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(walk_kernel, 4, sizeof(cl_mem), &walks_buffer);
        err |= clSetKernelArg(walk_kernel, 5, sizeof(int), &t);
        err |= clSetKernelArg(walk_kernel, 6, sizeof(int), &max_iter);
        err |= clSetKernelArg(walk_kernel, 7, sizeof(int), &dim);
        err |= clSetKernelArg(walk_kernel, 8, sizeof(int), &pop_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting walk kernel args: %d\n", err);
            free(fitness);
            goto cleanup;
        }

        global_work_size = pop_size;
        global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, walk_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing walk kernel: %d\n", err);
            free(fitness);
            goto cleanup;
        }

        // Update ant positions
        float elite_fitness = (float)opt->best_solution.fitness;
        float *elite_pos = (float *)malloc(dim * sizeof(float));
        if (!elite_pos) {
            fprintf(stderr, "Error: Memory allocation failed for elite_pos\n");
            free(fitness);
            goto cleanup;
        }
        for (int d = 0; d < dim; d++) {
            elite_pos[d] = (float)opt->best_solution.position[d];
        }

        err = clEnqueueWriteBuffer(opt->queue, elite_buffer, CL_TRUE, 0, 
                                  dim * sizeof(float), elite_pos, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(opt->queue, elite_fitness_buffer, CL_TRUE, 0, 
                                   sizeof(float), &elite_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing elite buffers: %d\n", err);
            free(fitness);
            free(elite_pos);
            goto cleanup;
        }

        err = clSetKernelArg(update_ants_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(update_ants_kernel, 1, sizeof(cl_mem), &antlion_buffer);
        err |= clSetKernelArg(update_ants_kernel, 2, sizeof(cl_mem), &elite_buffer);
        err |= clSetKernelArg(update_ants_kernel, 3, sizeof(cl_mem), &weights_buffer);
        err |= clSetKernelArg(update_ants_kernel, 4, sizeof(cl_mem), &cumsum_buffer);
        err |= clSetKernelArg(update_ants_kernel, 5, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(update_ants_kernel, 6, sizeof(int), &dim);
        err |= clSetKernelArg(update_ants_kernel, 7, sizeof(int), &pop_size);
        err |= clSetKernelArg(update_ants_kernel, 8, sizeof(int), &pop_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting update ants kernel args: %d\n", err);
            free(fitness);
            free(elite_pos);
            goto cleanup;
        }

        global_work_size = pop_size;
        global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, update_ants_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing update ants kernel: %d\n", err);
            free(fitness);
            free(elite_pos);
            goto cleanup;
        }

        // Enforce bound constraints on CPU
        enforce_bound_constraints(opt);

        // Update antlions
        err = clSetKernelArg(update_antlions_kernel, 0, sizeof(cl_mem), &antlion_buffer);
        err |= clSetKernelArg(update_antlions_kernel, 1, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(update_antlions_kernel, 2, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(update_antlions_kernel, 3, sizeof(cl_mem), &elite_buffer);
        err |= clSetKernelArg(update_antlions_kernel, 4, sizeof(cl_mem), &elite_fitness_buffer);
        err |= clSetKernelArg(update_antlions_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(update_antlions_kernel, 6, sizeof(int), &pop_size);
        err |= clSetKernelArg(update_antlions_kernel, 7, sizeof(int), &pop_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting update antlions kernel args: %d\n", err);
            free(fitness);
            free(elite_pos);
            goto cleanup;
        }

        global_work_size = pop_size;
        global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, update_antlions_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[3]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing update antlions kernel: %d\n", err);
            free(fitness);
            free(elite_pos);
            goto cleanup;
        }

        // Update elite on CPU
        double current_best_fitness = INFINITY;
        int current_best_idx = 0;
        for (int i = 0; i < pop_size; i++) {
            if (opt->population[i].fitness < current_best_fitness) {
                current_best_fitness = opt->population[i].fitness;
                current_best_idx = i;
            }
        }

        if (current_best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = current_best_fitness;
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = opt->population[current_best_idx].position[d];
            }
        }

        // Profiling
        cl_ulong time_start, time_end;
        double weights_time = 0.0, walk_time = 0.0, update_ants_time = 0.0, update_antlions_time = 0.0;
        for (int i = 0; i < 4; i++) {
            if (events[i]) {
                err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                    if (err == CL_SUCCESS) {
                        double time_ms = (time_end - time_start) / 1e6;
                        if (i == 0) weights_time = time_ms;
                        else if (i == 1) walk_time = time_ms;
                        else if (i == 2) update_ants_time = time_ms;
                        else if (i == 3) update_antlions_time = time_ms;
                    }
                }
            }
        }
        end_time = get_time_ms();
        printf("ALO|%5d -----> %9.16f | Total: %.3f ms | Weights: %.3f ms | Walk: %.3f ms | Update Ants: %.3f ms | Update Antlions: %.3f ms\n",
               t + 1, opt->best_solution.fitness, end_time - start_time, 
               weights_time, walk_time, update_ants_time, update_antlions_time);
        free(fitness);
        free(elite_pos);

        // Reset events
        for (int i = 0; i < 4; i++) {
            if (events[i]) {
                clReleaseEvent(events[i]);
                events[i] = 0;
            }
        }

        // Log progress every 50 iterations
        if ((t + 1) % 50 == 0) {
            printf("At iteration %d, the elite fitness is %f\n", t + 1, opt->best_solution.fitness);
        }
    }

    // Update CPU-side population
    float *final_positions = (float *)malloc(pop_size * dim * sizeof(float));
    if (!final_positions) {
        fprintf(stderr, "Error: Memory allocation failed for final_positions\n");
        goto cleanup;
    }
    err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                             pop_size * dim * sizeof(float), final_positions, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final population buffer: %d\n", err);
        free(final_positions);
        goto cleanup;
    }
    for (int i = 0; i < pop_size; i++) {
        for (int d = 0; d < dim; d++) {
            opt->population[i].position[d] = (double)final_positions[i * dim + d];
        }
    }
    free(final_positions);

cleanup:
    if (bounds) free(bounds);
    if (rng_states) free(rng_states);
    if (antlion_buffer) clReleaseMemObject(antlion_buffer);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (rng_states_buffer) clReleaseMemObject(rng_states_buffer);
    if (weights_buffer) clReleaseMemObject(weights_buffer);
    if (cumsum_buffer) clReleaseMemObject(cumsum_buffer);
    if (walks_buffer) clReleaseMemObject(walks_buffer);
    if (elite_buffer) clReleaseMemObject(elite_buffer);
    if (elite_fitness_buffer) clReleaseMemObject(elite_fitness_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (weights_kernel) clReleaseKernel(weights_kernel);
    if (walk_kernel) clReleaseKernel(walk_kernel);
    if (update_ants_kernel) clReleaseKernel(update_ants_kernel);
    if (update_antlions_kernel) clReleaseKernel(update_antlions_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 4; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (opt->queue) clFinish(opt->queue);
}
