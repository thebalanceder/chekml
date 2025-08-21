#include "ABC.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
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

// OpenCL kernel source for ABC
static const char *abc_kernel_source =
"#define ABC_ACCELERATION_BOUND 1.0f\n"
"#define ABC_ONLOOKER_RATIO 1.0f\n"
"#define ABC_TRIAL_LIMIT_FACTOR 0.6f\n"
// Random number generator
"uint lcg_rand(uint *seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float rand_float(uint *seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
// Initialize kernel
"__kernel void initialize_abc(\n"
"    __global float *positions,\n"
"    __global float *fitness,\n"
"    __global int *trial_counters,\n"
"    __global uint *rng_states,\n"
"    __global const float *bounds,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    __local float *local_bounds)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (idx >= pop_size) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    if (local_id < dim * 2) {\n"
"        local_bounds[local_id] = bounds[local_id];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (idx < pop_size) {\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float lower = local_bounds[d * 2];\n"
"            float upper = local_bounds[d * 2 + 1];\n"
"            positions[idx * dim + d] = lower + (upper - lower) * rand_float(&rng_state, 0.0f, 1.0f);\n"
"        }\n"
"        fitness[idx] = INFINITY;\n"
"        trial_counters[idx] = 0;\n"
"    }\n"
"    rng_states[idx] = rng_state;\n"
"}\n"
// Employed bee phase (computes new positions only)
"__kernel void employed_bee_phase(\n"
"    __global float *positions,\n"
"    __global int *trial_counters,\n"
"    __global uint *rng_states,\n"
"    __global const float *bounds,\n"
"    __global float *temp_positions,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    __local float *local_bounds)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (idx >= pop_size) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    if (local_id < dim * 2) {\n"
"        local_bounds[local_id] = bounds[local_id];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    int k = (int)(rand_float(&rng_state, 0.0f, (float)pop_size));\n"
"    if (k == idx) k = (k + 1) % pop_size;\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float phi = ABC_ACCELERATION_BOUND * rand_float(&rng_state, -1.0f, 1.0f);\n"
"        float new_pos = positions[idx * dim + d] + phi * (positions[idx * dim + d] - positions[k * dim + d]);\n"
"        float lower = local_bounds[d * 2];\n"
"        float upper = local_bounds[d * 2 + 1];\n"
"        new_pos = fmin(fmax(new_pos, lower), upper);\n"
"        temp_positions[idx * dim + d] = new_pos;\n"
"    }\n"
"    rng_states[idx] = rng_state;\n"
"}\n"
// Compute fitness probabilities
"__kernel void compute_probabilities(\n"
"    __global const float *fitness,\n"
"    __global float *probabilities,\n"
"    __global float *cumsum,\n"
"    const int pop_size,\n"
"    __local float *local_fitness)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    float sum = 0.0f;\n"
"    if (idx < pop_size) {\n"
"        local_fitness[local_id] = fitness[idx];\n"
"    } else {\n"
"        local_fitness[local_id] = 0.0f;\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int offset = local_size / 2; offset > 0; offset /= 2) {\n"
"        if (local_id < offset) {\n"
"            local_fitness[local_id] += local_fitness[local_id + offset];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (local_id == 0) {\n"
"        cumsum[0] = local_fitness[0];\n"
"    }\n"
"    barrier(CLK_GLOBAL_MEM_FENCE);\n"
"    float mean_fitness = cumsum[0] / pop_size;\n"
"    if (idx < pop_size) {\n"
"        probabilities[idx] = exp(-fitness[idx] / mean_fitness);\n"
"    }\n"
"    local_fitness[local_id] = idx < pop_size ? probabilities[idx] : 0.0f;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int offset = local_size / 2; offset > 0; offset /= 2) {\n"
"        if (local_id < offset) {\n"
"            local_fitness[local_id] += local_fitness[local_id + offset];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (local_id == 0) {\n"
"        cumsum[0] = local_fitness[0];\n"
"    }\n"
"    if (idx < pop_size) {\n"
"        probabilities[idx] = probabilities[idx] / cumsum[0];\n"
"        cumsum[idx] = probabilities[idx];\n"
"    }\n"
"    barrier(CLK_GLOBAL_MEM_FENCE);\n"
"    if (idx > 0 && idx < pop_size) {\n"
"        cumsum[idx] += cumsum[idx - 1];\n"
"    }\n"
"}\n"
// Onlooker bee phase (computes new positions only)
"__kernel void onlooker_bee_phase(\n"
"    __global float *positions,\n"
"    __global int *trial_counters,\n"
"    __global uint *rng_states,\n"
"    __global const float *probabilities,\n"
"    __global const float *cumsum,\n"
"    __global const float *bounds,\n"
"    __global float *temp_positions,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    const int n_onlookers,\n"
"    __local float *local_bounds)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (idx >= n_onlookers) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    if (local_id < dim * 2) {\n"
"        local_bounds[local_id] = bounds[local_id];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    float r = rand_float(&rng_state, 0.0f, 1.0f);\n"
"    int selected = 0;\n"
"    for (int i = 0; i < pop_size; i++) {\n"
"        if (r <= cumsum[i]) {\n"
"            selected = i;\n"
"            break;\n"
"        }\n"
"    }\n"
"    int k = (int)(rand_float(&rng_state, 0.0f, (float)pop_size));\n"
"    if (k == selected) k = (k + 1) % pop_size;\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float phi = ABC_ACCELERATION_BOUND * rand_float(&rng_state, -1.0f, 1.0f);\n"
"        float new_pos = positions[selected * dim + d] + phi * (positions[selected * dim + d] - positions[k * dim + d]);\n"
"        float lower = local_bounds[d * 2];\n"
"        float upper = local_bounds[d * 2 + 1];\n"
"        new_pos = fmin(fmax(new_pos, lower), upper);\n"
"        temp_positions[idx * dim + d] = new_pos;\n"
"    }\n"
"    rng_states[idx] = rng_state;\n"
"}\n"
// Scout bee phase
"__kernel void scout_bee_phase(\n"
"    __global float *positions,\n"
"    __global float *fitness,\n"
"    __global int *trial_counters,\n"
"    __global uint *rng_states,\n"
"    __global const float *bounds,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    const int trial_limit,\n"
"    __local float *local_bounds)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (idx >= pop_size) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    if (local_id < dim * 2) {\n"
"        local_bounds[local_id] = bounds[local_id];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (trial_counters[idx] >= trial_limit) {\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float lower = local_bounds[d * 2];\n"
"            float upper = local_bounds[d * 2 + 1];\n"
"            positions[idx * dim + d] = lower + (upper - lower) * rand_float(&rng_state, 0.0f, 1.0f);\n"
"        }\n"
"        fitness[idx] = INFINITY;\n"
"        trial_counters[idx] = 0;\n"
"    }\n"
"    rng_states[idx] = rng_state;\n"
"}\n"
// Find best solution
"__kernel void find_best(\n"
"    __global const float *fitness,\n"
"    __global int *best_idx,\n"
"    const int pop_size,\n"
"    __local float *local_fitness,\n"
"    __local int *local_indices)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    if (idx < pop_size) {\n"
"        local_fitness[local_id] = fitness[idx];\n"
"        local_indices[local_id] = idx;\n"
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
"        best_idx[get_group_id(0)] = local_indices[0];\n"
"    }\n"
"}\n";

// Main Optimization Function
void ABC_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, employed_kernel = NULL, prob_kernel = NULL;
    cl_kernel onlooker_kernel = NULL, scout_kernel = NULL, best_kernel = NULL;
    cl_command_queue queue = NULL;
    cl_mem positions_buffer = NULL, fitness_buffer = NULL, trial_counters_buffer = NULL;
    cl_mem rng_states_buffer = NULL, bounds_buffer = NULL, probabilities_buffer = NULL;
    cl_mem cumsum_buffer = NULL, best_idx_buffer = NULL, temp_positions_buffer = NULL;
    float *positions = NULL, *fitness = NULL, *bounds = NULL, *temp_positions = NULL;
    uint *rng_states = NULL;
    int *trial_counters = NULL;
    float *probabilities = NULL, *cumsum = NULL;
    double *cpu_position = NULL;
    cl_event events[6] = {0};
    double start_time, end_time;

    // Validate input
    if (!opt || !opt->population || !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure, null pointers, or missing objective function\n");
        goto cleanup;
    }
    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    const int max_iter = opt->max_iter;
    if (dim < 1 || pop_size < 1 || max_iter < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), pop_size (%d), or max_iter (%d)\n", 
                dim, pop_size, max_iter);
        goto cleanup;
    }
    if (dim > 32) {
        fprintf(stderr, "Error: dim (%d) exceeds maximum supported value of 32\n", dim);
        goto cleanup;
    }
    for (int d = 0; d < dim; d++) {
        if (opt->bounds[d * 2] >= opt->bounds[d * 2 + 1]) {
            fprintf(stderr, "Error: Invalid bounds for dimension %d: [%f, %f]\n", 
                    d, opt->bounds[d * 2], opt->bounds[d * 2 + 1]);
            goto cleanup;
        }
    }

    // Initialize OpenCL program
    program = clCreateProgramWithSource(opt->context, 1, &abc_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating ABC program: %d\n", err);
        goto cleanup;
    }
    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building ABC program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        goto cleanup;
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_abc", &err);
    employed_kernel = clCreateKernel(program, "employed_bee_phase", &err);
    prob_kernel = clCreateKernel(program, "compute_probabilities", &err);
    onlooker_kernel = clCreateKernel(program, "onlooker_bee_phase", &err);
    scout_kernel = clCreateKernel(program, "scout_bee_phase", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating ABC kernels: %d\n", err);
        goto cleanup;
    }

    // Create command queue with profiling
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue = clCreateCommandQueueWithProperties(opt->context, opt->device, props, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating command queue: %d\n", err);
        goto cleanup;
    }

    // Query device capabilities
    size_t max_work_group_size;
    err = clGetDeviceInfo(opt->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying CL_DEVICE_MAX_WORK_GROUP_SIZE: %d\n", err);
        goto cleanup;
    }
    size_t local_work_size = max_work_group_size < 64 ? max_work_group_size : 64;

    // Allocate GPU buffers
    positions_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(float), NULL, &err);
    trial_counters_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(int), NULL, &err);
    rng_states_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(uint), NULL, &err);
    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, dim * 2 * sizeof(float), NULL, &err);
    probabilities_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(float), NULL, &err);
    cumsum_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(int), NULL, &err);
    temp_positions_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating ABC buffers: %d\n", err);
        goto cleanup;
    }

    // Allocate host memory
    positions = (float *)malloc(pop_size * dim * sizeof(float));
    fitness = (float *)malloc(pop_size * sizeof(float));
    bounds = (float *)malloc(dim * 2 * sizeof(float));
    trial_counters = (int *)malloc(pop_size * sizeof(int));
    rng_states = (uint *)malloc(pop_size * sizeof(uint));
    probabilities = (float *)malloc(pop_size * sizeof(float));
    cumsum = (float *)malloc(pop_size * sizeof(float));
    temp_positions = (float *)malloc(pop_size * dim * sizeof(float));
    cpu_position = (double *)malloc(dim * sizeof(double));
    if (!positions || !fitness || !bounds || !trial_counters || !rng_states || 
        !probabilities || !cumsum || !temp_positions || !cpu_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }

    // Initialize host data
    for (int d = 0; d < dim; d++) {
        bounds[d * 2] = (float)opt->bounds[d * 2];
        bounds[d * 2 + 1] = (float)opt->bounds[d * 2 + 1];
    }
    for (int i = 0; i < pop_size; i++) {
        rng_states[i] = (uint)(time(NULL) ^ (i + 1));
        trial_counters[i] = 0;
        for (int d = 0; d < dim; d++) {
            positions[i * dim + d] = (float)opt->population[i].position[d];
        }
        fitness[i] = INFINITY;
    }

    // Write initial data to GPU
    err = clEnqueueWriteBuffer(queue, positions_buffer, CL_TRUE, 0, 
                              pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(float), fitness, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, trial_counters_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(int), trial_counters, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, rng_states_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(uint), rng_states, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 
                               dim * 2 * sizeof(float), bounds, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial buffers: %d\n", err);
        goto cleanup;
    }

    // Initialize population
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &positions_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &fitness_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(cl_mem), &trial_counters_buffer);
    err |= clSetKernelArg(init_kernel, 3, sizeof(cl_mem), &rng_states_buffer);
    err |= clSetKernelArg(init_kernel, 4, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 5, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 6, sizeof(int), &pop_size);
    err |= clSetKernelArg(init_kernel, 7, 2 * dim * sizeof(float), NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting init kernel args: %d\n", err);
        goto cleanup;
    }
    size_t global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[0]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel: %d\n", err);
        goto cleanup;
    }

    // Evaluate initial fitness on CPU
    err = clEnqueueReadBuffer(queue, positions_buffer, CL_TRUE, 0, 
                             pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading initial positions: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < pop_size; i++) {
        for (int d = 0; d < dim; d++) {
            cpu_position[d] = (double)positions[i * dim + d];
        }
        fitness[i] = (float)objective_function(cpu_position);
        if (!isfinite(fitness[i])) {
            fprintf(stderr, "Warning: Non-finite fitness for bee %d: %f\n", i, fitness[i]);
            fitness[i] = INFINITY;
        }
    }
    err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                              pop_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial fitness: %d\n", err);
        goto cleanup;
    }

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < pop_size; i++) {
        if (fitness[i] < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)fitness[i];
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)positions[i * dim + d];
            }
        }
    }

    // Main optimization loop
    int n_onlookers = (int)(ABC_ONLOOKER_RATIO * pop_size);
    int trial_limit = (int)(ABC_TRIAL_LIMIT_FACTOR * dim * pop_size);
    float *new_fitness = (float *)malloc(pop_size * sizeof(float));
    if (!new_fitness) {
        fprintf(stderr, "Error: Memory allocation failed for new_fitness\n");
        goto cleanup;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Employed bee phase
        err = clSetKernelArg(employed_kernel, 0, sizeof(cl_mem), &positions_buffer);
        err |= clSetKernelArg(employed_kernel, 1, sizeof(cl_mem), &trial_counters_buffer);
        err |= clSetKernelArg(employed_kernel, 2, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(employed_kernel, 3, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(employed_kernel, 4, sizeof(cl_mem), &temp_positions_buffer);
        err |= clSetKernelArg(employed_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(employed_kernel, 6, sizeof(int), &pop_size);
        err |= clSetKernelArg(employed_kernel, 7, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting employed kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, employed_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing employed kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate new positions on CPU
        err = clEnqueueReadBuffer(queue, temp_positions_buffer, CL_TRUE, 0, 
                                 pop_size * dim * sizeof(float), temp_positions, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading temp_positions: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < pop_size; i++) {
            for (int d = 0; d < dim; d++) {
                cpu_position[d] = (double)temp_positions[i * dim + d];
            }
            new_fitness[i] = (float)objective_function(cpu_position);
            if (!isfinite(new_fitness[i])) {
                fprintf(stderr, "Warning: Non-finite fitness for bee %d: %f\n", i, new_fitness[i]);
                new_fitness[i] = INFINITY;
            }
        }

        // Greedy selection on CPU
        err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                                 pop_size * sizeof(float), fitness, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(queue, trial_counters_buffer, CL_TRUE, 0, 
                                  pop_size * sizeof(int), trial_counters, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading fitness/trial_counters: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < pop_size; i++) {
            if (new_fitness[i] <= fitness[i]) {
                for (int d = 0; d < dim; d++) {
                    positions[i * dim + d] = temp_positions[i * dim + d];
                }
                fitness[i] = new_fitness[i];
                trial_counters[i] = 0;
            } else {
                trial_counters[i]++;
            }
        }
        err = clEnqueueWriteBuffer(queue, positions_buffer, CL_TRUE, 0, 
                                  pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                                   pop_size * sizeof(float), fitness, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, trial_counters_buffer, CL_TRUE, 0, 
                                   pop_size * sizeof(int), trial_counters, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing positions/fitness/trial_counters: %d\n", err);
            goto cleanup;
        }

        // Compute probabilities
        err = clSetKernelArg(prob_kernel, 0, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(prob_kernel, 1, sizeof(cl_mem), &probabilities_buffer);
        err |= clSetKernelArg(prob_kernel, 2, sizeof(cl_mem), &cumsum_buffer);
        err |= clSetKernelArg(prob_kernel, 3, sizeof(int), &pop_size);
        err |= clSetKernelArg(prob_kernel, 4, local_work_size * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting probabilities kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, prob_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing probabilities kernel: %d\n", err);
            goto cleanup;
        }

        // Onlooker bee phase
        err = clSetKernelArg(onlooker_kernel, 0, sizeof(cl_mem), &positions_buffer);
        err |= clSetKernelArg(onlooker_kernel, 1, sizeof(cl_mem), &trial_counters_buffer);
        err |= clSetKernelArg(onlooker_kernel, 2, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(onlooker_kernel, 3, sizeof(cl_mem), &probabilities_buffer);
        err |= clSetKernelArg(onlooker_kernel, 4, sizeof(cl_mem), &cumsum_buffer);
        err |= clSetKernelArg(onlooker_kernel, 5, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(onlooker_kernel, 6, sizeof(cl_mem), &temp_positions_buffer);
        err |= clSetKernelArg(onlooker_kernel, 7, sizeof(int), &dim);
        err |= clSetKernelArg(onlooker_kernel, 8, sizeof(int), &pop_size);
        err |= clSetKernelArg(onlooker_kernel, 9, sizeof(int), &n_onlookers);
        err |= clSetKernelArg(onlooker_kernel, 10, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting onlooker kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((n_onlookers + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, onlooker_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[3]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing onlooker kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate onlooker positions on CPU
        err = clEnqueueReadBuffer(queue, temp_positions_buffer, CL_TRUE, 0, 
                                 n_onlookers * dim * sizeof(float), temp_positions, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading onlooker temp_positions: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < n_onlookers; i++) {
            for (int d = 0; d < dim; d++) {
                cpu_position[d] = (double)temp_positions[i * dim + d];
            }
            new_fitness[i] = (float)objective_function(cpu_position);
            if (!isfinite(new_fitness[i])) {
                fprintf(stderr, "Warning: Non-finite fitness for onlooker %d: %f\n", i, new_fitness[i]);
                new_fitness[i] = INFINITY;
            }
        }

        // Greedy selection for onlookers on CPU
        err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                                 pop_size * sizeof(float), fitness, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(queue, trial_counters_buffer, CL_TRUE, 0, 
                                  pop_size * sizeof(int), trial_counters, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(queue, cumsum_buffer, CL_TRUE, 0, 
                                  pop_size * sizeof(float), cumsum, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading fitness/trial_counters/cumsum: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < n_onlookers; i++) {
            // Roulette wheel selection on CPU
            float r = (float)rand() / RAND_MAX;
            int selected = 0;
            for (int j = 0; j < pop_size; j++) {
                if (r <= cumsum[j]) {
                    selected = j;
                    break;
                }
            }
            if (new_fitness[i] <= fitness[selected]) {
                for (int d = 0; d < dim; d++) {
                    positions[selected * dim + d] = temp_positions[i * dim + d];
                }
                fitness[selected] = new_fitness[i];
                trial_counters[selected] = 0;
            } else {
                trial_counters[selected]++;
            }
        }
        err = clEnqueueWriteBuffer(queue, positions_buffer, CL_TRUE, 0, 
                                  pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                                   pop_size * sizeof(float), fitness, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, trial_counters_buffer, CL_TRUE, 0, 
                                   pop_size * sizeof(int), trial_counters, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing onlooker positions/fitness/trial_counters: %d\n", err);
            goto cleanup;
        }

        // Scout bee phase
        err = clSetKernelArg(scout_kernel, 0, sizeof(cl_mem), &positions_buffer);
        err |= clSetKernelArg(scout_kernel, 1, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(scout_kernel, 2, sizeof(cl_mem), &trial_counters_buffer);
        err |= clSetKernelArg(scout_kernel, 3, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(scout_kernel, 4, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(scout_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(scout_kernel, 6, sizeof(int), &pop_size);
        err |= clSetKernelArg(scout_kernel, 7, sizeof(int), &trial_limit);
        err |= clSetKernelArg(scout_kernel, 8, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting scout kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, scout_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[4]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing scout kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate scout positions on CPU
        err = clEnqueueReadBuffer(queue, positions_buffer, CL_TRUE, 0, 
                                 pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                                  pop_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading scout positions/fitness: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < pop_size; i++) {
            if (fitness[i] == INFINITY) {
                for (int d = 0; d < dim; d++) {
                    cpu_position[d] = (double)positions[i * dim + d];
                }
                fitness[i] = (float)objective_function(cpu_position);
                if (!isfinite(fitness[i])) {
                    fprintf(stderr, "Warning: Non-finite fitness for scout %d: %f\n", i, fitness[i]);
                    fitness[i] = INFINITY;
                }
            }
        }
        err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                                  pop_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing scout fitness: %d\n", err);
            goto cleanup;
        }

        // Find best solution
        err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &best_idx_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(int), &pop_size);
        err |= clSetKernelArg(best_kernel, 3, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 4, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[5]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing best kernel: %d\n", err);
            goto cleanup;
        }

        // Update best solution
        int best_idx;
        err = clEnqueueReadBuffer(queue, best_idx_buffer, CL_TRUE, 0, sizeof(int), &best_idx, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best_idx buffer: %d\n", err);
            goto cleanup;
        }
        if (best_idx >= 0 && best_idx < pop_size) {
            float best_fitness_check;
            err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, best_idx * sizeof(float), 
                                     sizeof(float), &best_fitness_check, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best fitness: %d\n", err);
                goto cleanup;
            }
            if (isfinite(best_fitness_check) && best_fitness_check < opt->best_solution.fitness) {
                opt->best_solution.fitness = (double)best_fitness_check;
                float *best_solution = (float *)malloc(dim * sizeof(float));
                if (!best_solution) {
                    fprintf(stderr, "Error: Memory allocation failed for best_solution\n");
                    goto cleanup;
                }
                err = clEnqueueReadBuffer(queue, positions_buffer, CL_TRUE, best_idx * dim * sizeof(float), 
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
        } else {
            fprintf(stderr, "Warning: Invalid best_idx %d at iteration %d\n", best_idx, iter + 1);
        }

        // Profiling
        err = clFinish(queue);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error finishing queue: %d\n", err);
            goto cleanup;
        }
        cl_ulong time_start, time_end;
        double init_time = 0.0, employed_time = 0.0, prob_time = 0.0, onlooker_time = 0.0, scout_time = 0.0, best_time = 0.0;
        for (int i = 0; i < 6; i++) {
            if (events[i]) {
                err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                err |= clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                if (err == CL_SUCCESS) {
                    double time_ms = (time_end - time_start) / 1e6;
                    if (i == 0) init_time = time_ms;
                    else if (i == 1) employed_time = time_ms;
                    else if (i == 2) prob_time = time_ms;
                    else if (i == 3) onlooker_time = time_ms;
                    else if (i == 4) scout_time = time_ms;
                    else if (i == 5) best_time = time_ms;
                }
            }
        }
        end_time = get_time_ms();
        printf("ABC|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Employed: %.3f ms | Prob: %.3f ms | Onlooker: %.3f ms | Scout: %.3f ms | Best: %.3f ms\n",
               iter + 1, opt->best_solution.fitness, end_time - start_time, 
               init_time, employed_time, prob_time, onlooker_time, scout_time, best_time);

        // Reset events
        for (int i = 0; i < 6; i++) {
            if (events[i]) {
                clReleaseEvent(events[i]);
                events[i] = 0;
            }
        }
    }

    // Update CPU-side population
    err = clEnqueueReadBuffer(queue, positions_buffer, CL_TRUE, 0, 
                             pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                              pop_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final buffers: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < pop_size; i++) {
        for (int d = 0; d < dim; d++) {
            float p = positions[i * dim + d];
            float lower_bound = bounds[d * 2];
            float upper_bound = bounds[d * 2 + 1];
            if (p < lower_bound || p > upper_bound || !isfinite(p)) {
                fprintf(stderr, "Warning: Final position out of bounds or non-finite for bee %d, dim %d: %f\n", i, d, p);
                p = lower_bound + (upper_bound - lower_bound) * ((float)rand() / RAND_MAX);
                positions[i * dim + d] = p;
            }
            opt->population[i].position[d] = (double)p;
        }
        opt->population[i].fitness = (double)fitness[i];
    }

cleanup:
    if (positions) free(positions);
    if (fitness) free(fitness);
    if (bounds) free(bounds);
    if (trial_counters) free(trial_counters);
    if (rng_states) free(rng_states);
    if (probabilities) free(probabilities);
    if (cumsum) free(cumsum);
    if (temp_positions) free(temp_positions);
    if (cpu_position) free(cpu_position);
    if (new_fitness) free(new_fitness);
    if (positions_buffer) clReleaseMemObject(positions_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (trial_counters_buffer) clReleaseMemObject(trial_counters_buffer);
    if (rng_states_buffer) clReleaseMemObject(rng_states_buffer);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (probabilities_buffer) clReleaseMemObject(probabilities_buffer);
    if (cumsum_buffer) clReleaseMemObject(cumsum_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (temp_positions_buffer) clReleaseMemObject(temp_positions_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (employed_kernel) clReleaseKernel(employed_kernel);
    if (prob_kernel) clReleaseKernel(prob_kernel);
    if (onlooker_kernel) clReleaseKernel(onlooker_kernel);
    if (scout_kernel) clReleaseKernel(scout_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 6; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) {
        clFinish(queue);
        clReleaseCommandQueue(queue);
    }
}
