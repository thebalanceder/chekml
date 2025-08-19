#include "ACO.h"
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

// OpenCL kernel source for ACO
static const char *aco_kernel_source = 
"#define ACO_N_BINS 10\n"
"#define ACO_Q 1.0f\n"
"#define ACO_RHO 0.1f\n"
// Helper function for atomic float addition
"void atomic_add_float(__global float *addr, float val) {\n"
"    union {\n"
"        uint u;\n"
"        float f;\n"
"    } old_val, new_val;\n"
"    do {\n"
"        old_val.f = *addr;\n"
"        new_val.f = old_val.f + val;\n"
"    } while (atomic_cmpxchg((volatile __global uint *)addr, old_val.u, new_val.u) != old_val.u);\n"
"}\n"
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
// Roulette wheel selection
"int roulette_wheel_selection(__private float *prob, int size, uint *rng_state) {\n"
"    float r = rand_float(rng_state, 0.0f, 1.0f);\n"
"    float cumsum = 0.0f;\n"
"    for (int i = 0; i < size; i++) {\n"
"        cumsum += prob[i];\n"
"        if (r <= cumsum) {\n"
"            return i;\n"
"        }\n"
"    }\n"
"    return size - 1;\n"
"}\n"
// Kernel to construct solutions
"__kernel void construct_solutions(\n"
"    __global float *tau,\n"
"    __global float *bins,\n"
"    __global int *tours,\n"
"    __global float *positions,\n"
"    __global uint *rng_states,\n"
"    const int dim,\n"
"    const int n_bins,\n"
"    const int n_ants)\n"
"{\n"
"    int ant_idx = get_global_id(0);\n"
"    if (ant_idx >= n_ants) return;\n"
"    uint rng_state = rng_states[ant_idx];\n"
"    float prob[ACO_N_BINS];\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float sum_P = 0.0f;\n"
"        for (int i = 0; i < n_bins; i++) {\n"
"            prob[i] = tau[i * dim + d];\n"
"            sum_P += prob[i];\n"
"        }\n"
"        if (sum_P > 0.0f) {\n"
"            float inv_sum_P = 1.0f / sum_P;\n"
"            for (int i = 0; i < n_bins; i++) {\n"
"                prob[i] *= inv_sum_P;\n"
"            }\n"
"        }\n"
"        int bin_idx = roulette_wheel_selection(prob, n_bins, &rng_state);\n"
"        tours[ant_idx * dim + d] = bin_idx;\n"
"        positions[ant_idx * dim + d] = bins[bin_idx * dim + d];\n"
"    }\n"
"    rng_states[ant_idx] = rng_state;\n"
"}\n"
// Kernel to update pheromones
"__kernel void update_pheromones(\n"
"    __global float *tau,\n"
"    __global int *tours,\n"
"    __global float *fitness,\n"
"    const float best_fitness,\n"
"    const int dim,\n"
"    const int n_ants)\n"
"{\n"
"    int ant_idx = get_global_id(0);\n"
"    if (ant_idx >= n_ants) return;\n"
"    float delta = ACO_Q / (1.0f + max(0.0f, fitness[ant_idx] - best_fitness));\n"
"    for (int d = 0; d < dim; d++) {\n"
"        int bin_idx = tours[ant_idx * dim + d];\n"
"        atomic_add_float(&tau[bin_idx * dim + d], delta);\n"
"    }\n"
"}\n"
// Kernel to evaporate pheromones
"__kernel void evaporate_pheromones(\n"
"    __global float *tau,\n"
"    const int dim,\n"
"    const int n_bins)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= n_bins * dim) return;\n"
"    tau[idx] *= (1.0f - ACO_RHO);\n"
"}\n";

// Main Optimization Function
void ACO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
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
    const int population_size = ACO_N_ANT;
    const int max_iter = ACO_MAX_ITER;
    if (dim < 1 || population_size < 1 || max_iter < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), or max_iter (%d)\n", 
                dim, population_size, max_iter);
        return;
    }

    // Initialize OpenCL program and kernels
    cl_program program = clCreateProgramWithSource(opt->context, 1, &aco_kernel_source, NULL, &err);
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

    cl_kernel construct_kernel = clCreateKernel(program, "construct_solutions", &err);
    cl_kernel update_kernel = clCreateKernel(program, "update_pheromones", &err);
    cl_kernel evaporate_kernel = clCreateKernel(program, "evaporate_pheromones", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating kernels: %d\n", err);
        clReleaseProgram(program);
        return;
    }

    // Allocate GPU buffers
    cl_mem tau_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, 
                                       ACO_N_BINS * dim * sizeof(float), NULL, &err);
    cl_mem bins_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, 
                                        ACO_N_BINS * dim * sizeof(float), NULL, &err);
    cl_mem tours_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, 
                                         ACO_N_ANT * dim * sizeof(int), NULL, &err);
    cl_mem rng_states_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, 
                                              ACO_N_ANT * sizeof(uint), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffers: %d\n", err);
        clReleaseKernel(construct_kernel);
        clReleaseKernel(update_kernel);
        clReleaseKernel(evaporate_kernel);
        clReleaseProgram(program);
        return;
    }

    // Initialize tau and bins on host
    float *tau = (float *)malloc(ACO_N_BINS * dim * sizeof(float));
    float *bins = (float *)malloc(ACO_N_BINS * dim * sizeof(float));
    uint *rng_states = (uint *)malloc(ACO_N_ANT * sizeof(uint));
    if (!tau || !bins || !rng_states) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }

    for (int d = 0; d < dim; d++) {
        float lower = (float)opt->bounds[2 * d];
        float range = (float)(opt->bounds[2 * d + 1] - lower);
        float bin_step = range / (ACO_N_BINS - 1);
        for (int i = 0; i < ACO_N_BINS; i++) {
            tau[i * dim + d] = ACO_TAU0;
            bins[i * dim + d] = lower + i * bin_step;
        }
    }
    for (int i = 0; i < ACO_N_ANT; i++) {
        rng_states[i] = (uint)(i + 1) * 12345U; // Simple seed
    }

    // Write initial data to GPU
    err = clEnqueueWriteBuffer(opt->queue, tau_buffer, CL_TRUE, 0, 
                               ACO_N_BINS * dim * sizeof(float), tau, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(opt->queue, bins_buffer, CL_TRUE, 0, 
                                ACO_N_BINS * dim * sizeof(float), bins, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(opt->queue, rng_states_buffer, CL_TRUE, 0, 
                                ACO_N_ANT * sizeof(uint), rng_states, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing buffers: %d\n", err);
        goto cleanup;
    }

    // Set population size and max iterations
    opt->population_size = ACO_N_ANT;
    opt->max_iter = ACO_MAX_ITER;

    // Profiling setup
    cl_event events[3] = {0}; // Events for construct, update, evaporate
    double start_time, end_time;

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        start_time = get_time_ms();

        // Set kernel arguments for construct_solutions
        int n_bins = ACO_N_BINS;
        int n_ants = ACO_N_ANT;
        err = clSetKernelArg(construct_kernel, 0, sizeof(cl_mem), &tau_buffer);
        err |= clSetKernelArg(construct_kernel, 1, sizeof(cl_mem), &bins_buffer);
        err |= clSetKernelArg(construct_kernel, 2, sizeof(cl_mem), &tours_buffer);
        err |= clSetKernelArg(construct_kernel, 3, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(construct_kernel, 4, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(construct_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(construct_kernel, 6, sizeof(int), &n_bins);
        err |= clSetKernelArg(construct_kernel, 7, sizeof(int), &n_ants);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting construct kernel args: %d\n", err);
            goto cleanup;
        }

        // Execute construct_solutions kernel
        size_t global_work_size = ACO_N_ANT;
        size_t local_work_size = 64;
        global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, construct_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[0]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing construct kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate fitness on CPU (since objective_function is a CPU callback)
        float *positions = (float *)malloc(population_size * dim * sizeof(float));
        if (!positions) {
            fprintf(stderr, "Error: Memory allocation failed for positions\n");
            goto cleanup;
        }
        err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                                  population_size * dim * sizeof(float), positions, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading population buffer: %d\n", err);
            free(positions);
            goto cleanup;
        }

        float *fitness = (float *)malloc(population_size * sizeof(float));
        if (!fitness) {
            fprintf(stderr, "Error: Memory allocation failed for fitness\n");
            free(positions);
            goto cleanup;
        }
        for (int k = 0; k < population_size; k++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            if (!pos) {
                fprintf(stderr, "Error: Memory allocation failed for pos\n");
                free(fitness);
                free(positions);
                goto cleanup;
            }
            for (int d = 0; d < dim; d++) {
                pos[d] = (double)positions[k * dim + d];
            }
            opt->population[k].fitness = objective_function(pos);
            fitness[k] = (float)opt->population[k].fitness;
            free(pos);
        }
        free(positions);

        // Write fitness to GPU
        err = clEnqueueWriteBuffer(opt->queue, opt->fitness_buffer, CL_TRUE, 0, 
                                   population_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            free(fitness);
            goto cleanup;
        }

        // Enforce bound constraints on CPU
        enforce_bound_constraints(opt);

        // Update pheromones
        float best_fitness = (float)opt->best_solution.fitness;
        err = clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &tau_buffer);
        err |= clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &tours_buffer);
        err |= clSetKernelArg(update_kernel, 2, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(update_kernel, 3, sizeof(float), &best_fitness);
        err |= clSetKernelArg(update_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(update_kernel, 5, sizeof(int), &n_ants);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting update kernel args: %d\n", err);
            free(fitness);
            goto cleanup;
        }

        global_work_size = ACO_N_ANT;
        global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, update_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing update kernel: %d\n", err);
            free(fitness);
            goto cleanup;
        }

        // Evaporate pheromones
        err = clSetKernelArg(evaporate_kernel, 0, sizeof(cl_mem), &tau_buffer);
        err |= clSetKernelArg(evaporate_kernel, 1, sizeof(int), &dim);
        err |= clSetKernelArg(evaporate_kernel, 2, sizeof(int), &n_bins);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting evaporate kernel args: %d\n", err);
            free(fitness);
            goto cleanup;
        }

        global_work_size = ACO_N_BINS * dim;
        global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(opt->queue, evaporate_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing evaporate kernel: %d\n", err);
            free(fitness);
            goto cleanup;
        }

        // Find best solution on CPU
        double current_best_fitness = INFINITY;
        int current_best_idx = 0;
        for (int k = 0; k < population_size; k++) {
            if (opt->population[k].fitness < current_best_fitness) {
                current_best_fitness = opt->population[k].fitness;
                current_best_idx = k;
            }
        }

        // Update global best
        if (current_best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = current_best_fitness;
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = opt->population[current_best_idx].position[d];
            }
        }

        // Profiling
        cl_ulong time_start, time_end;
        double construct_time = 0.0, update_time = 0.0, evaporate_time = 0.0;
        for (int i = 0; i < 3; i++) {
            if (events[i]) {
                err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                    if (err == CL_SUCCESS) {
                        double time_ms = (time_end - time_start) / 1e6;
                        if (i == 0) construct_time = time_ms;
                        else if (i == 1) update_time = time_ms;
                        else if (i == 2) evaporate_time = time_ms;
                    }
                }
            }
        }
        end_time = get_time_ms();
        printf("ACO|%5d -----> %9.16f | Total: %.3f ms | Construct: %.3f ms | Update: %.3f ms | Evaporate: %.3f ms\n",
               iter + 1, opt->best_solution.fitness, end_time - start_time, 
               construct_time, update_time, evaporate_time);
        free(fitness);

        // Reset events for the next iteration
        for (int i = 0; i < 3; i++) {
            if (events[i]) {
                clReleaseEvent(events[i]);
                events[i] = 0;
            }
        }
    }

    // Update CPU-side population
    float *final_positions = (float *)malloc(population_size * dim * sizeof(float));
    if (!final_positions) {
        fprintf(stderr, "Error: Memory allocation failed for final_positions\n");
        goto cleanup;
    }
    err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                              population_size * dim * sizeof(float), final_positions, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final population buffer: %d\n", err);
        free(final_positions);
        goto cleanup;
    }
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            opt->population[i].position[d] = (double)final_positions[i * dim + d];
        }
    }
    free(final_positions);

cleanup:
    // Free allocated memory
    if (tau) free(tau);
    if (bins) free(bins);
    if (rng_states) free(rng_states);
    if (tau_buffer) clReleaseMemObject(tau_buffer);
    if (bins_buffer) clReleaseMemObject(bins_buffer);
    if (tours_buffer) clReleaseMemObject(tours_buffer);
    if (rng_states_buffer) clReleaseMemObject(rng_states_buffer);
    if (construct_kernel) clReleaseKernel(construct_kernel);
    if (update_kernel) clReleaseKernel(update_kernel);
    if (evaporate_kernel) clReleaseKernel(evaporate_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 3; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (opt->queue) clFinish(opt->queue);
}
