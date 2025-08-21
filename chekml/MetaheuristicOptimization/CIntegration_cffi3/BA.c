#include "BA.h"
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

// Define clamp macro for C code
#define clamp(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

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

// OpenCL kernel source for BA
static const char *ba_kernel_source =
"#define LOUDNESS 1.0f\n"
"#define PULSE_RATE 1.0f\n"
"#define ALPHA_BA 0.97f\n"
"#define GAMMA 0.1f\n"
"#define FREQ_MIN 0.0f\n"
"#define FREQ_MAX 2.0f\n"
"#define LOCAL_SEARCH_SCALE 0.1f\n"
"#define VELOCITY_MAX 10.0f\n"
"#define POSITION_MAX 1000.0f\n"
// Define clamp macro for OpenCL
"#define clamp(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))\n"
// LCG random number generator (from CS)
"uint lcg_rand(uint *seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float rand_float(uint *seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
// Kernel to initialize fitness
"__kernel void initialize_fitness(\n"
"    __global float *fitness,\n"
"    const int pop_size)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    if (idx < pop_size) {\n"
"        fitness[idx] = INFINITY;\n"
"    }\n"
"}\n"
// Kernel for frequency update and velocity adjustment
"__kernel void frequency_update(\n"
"    __global float *positions,\n"
"    __global float *velocities,\n"
"    __global float *freq,\n"
"    __global const float *best_pos,\n"
"    __global uint *rng_states,\n"
"    __global const float *bounds,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    const int iter,\n"
"    const int max_iter,\n"
"    __local float *local_bounds)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (idx >= pop_size) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    // Cache bounds in local memory\n"
"    if (local_id < dim * 2) {\n"
"        local_bounds[local_id] = bounds[local_id];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    // Update frequency\n"
"    float freq_range = FREQ_MAX - FREQ_MIN;\n"
"    freq[idx] = FREQ_MIN + freq_range * rand_float(&rng_state, 0.0f, 1.0f);\n"
"    // Update velocity and position\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float pos = positions[idx * dim + d];\n"
"        float vel = velocities[idx * dim + d];\n"
"        float delta = pos - best_pos[d];\n"
"        if (isinf(delta) || isnan(delta)) delta = 0.0f;\n"
"        vel += delta * freq[idx];\n"
"        float lower_bound = local_bounds[d * 2];\n"
"        float upper_bound = local_bounds[d * 2 + 1];\n"
"        float domain_size = upper_bound - lower_bound;\n"
"        vel = clamp(vel, -0.1f * domain_size, 0.1f * domain_size);\n"
"        pos += vel;\n"
"        if (pos <= lower_bound || pos >= upper_bound) {\n"
"            pos = lower_bound + (upper_bound - lower_bound) * rand_float(&rng_state, 0.0f, 0.1f);\n"
"        } else {\n"
"            pos = clamp(pos, lower_bound, upper_bound);\n"
"        }\n"
"        if (isinf(pos) || isnan(pos)) pos = lower_bound;\n"
"        positions[idx * dim + d] = pos;\n"
"        velocities[idx * dim + d] = vel;\n"
"    }\n"
"    rng_states[idx] = rng_state;\n"
"}\n"
// Kernel for local search
"__kernel void local_search(\n"
"    __global float *positions,\n"
"    __global const float *best_pos,\n"
"    __global uint *rng_states,\n"
"    __global const float *bounds,\n"
"    const float pulse_rate,\n"
"    const float loudness,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    const int iter,\n"
"    const int max_iter,\n"
"    __local float *local_bounds)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (idx >= pop_size) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    // Cache bounds in local memory\n"
"    if (local_id < dim * 2) {\n"
"        local_bounds[local_id] = bounds[local_id];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (rand_float(&rng_state, 0.0f, 1.0f) < pulse_rate) {\n"
"        float scale = LOCAL_SEARCH_SCALE * loudness * (1.0f + 0.5f * ((float)iter / (float)max_iter));\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float delta = 2.0f * rand_float(&rng_state, 0.0f, 1.0f) - 1.0f;\n"
"            float pos = best_pos[d] + scale * delta;\n"
"            float lower_bound = local_bounds[d * 2];\n"
"            float upper_bound = local_bounds[d * 2 + 1];\n"
"            if (pos <= lower_bound || pos >= upper_bound) {\n"
"                pos = lower_bound + (upper_bound - lower_bound) * rand_float(&rng_state, 0.0f, 0.1f);\n"
"            } else {\n"
"                pos = clamp(pos, lower_bound, upper_bound);\n"
"            }\n"
"            if (isinf(pos) || isnan(pos)) pos = lower_bound;\n"
"            positions[idx * dim + d] = pos;\n"
"        }\n"
"    }\n"
"    rng_states[idx] = rng_state;\n"
"}\n"
// Kernel to find best solution
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
void BA_optimize(Optimizer *restrict opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, freq_kernel = NULL, local_kernel = NULL, best_kernel = NULL;
    cl_command_queue queue = NULL;
    cl_mem freq_buffer = NULL, velocities_buffer = NULL, best_pos_buffer = NULL;
    cl_mem best_fitness_buffer = NULL, rng_states_buffer = NULL, bounds_buffer = NULL;
    cl_mem best_idx_buffer = NULL;
    float *positions = NULL, *fitness = NULL, *best_pos = NULL, *bounds = NULL;
    float *velocities = NULL;
    float *freq = NULL;
    uint *rng_states = NULL;
    int *best_idx_array = NULL;
    cl_event events[4] = {0};
    double start_time, end_time;
    double *temp_position = NULL;

    // Validate input
    if (!opt || !opt->population || !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        goto cleanup;
    }
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: Population[%d].position is null\n", i);
            goto cleanup;
        }
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

    // Validate bounds
    for (int d = 0; d < dim; d++) {
        if (opt->bounds[d * 2] >= opt->bounds[d * 2 + 1]) {
            fprintf(stderr, "Error: Invalid bounds for dimension %d: [%f, %f]\n", 
                    d, opt->bounds[d * 2], opt->bounds[d * 2 + 1]);
            goto cleanup;
        }
    }

    // Initialize OpenCL program and kernels
    program = clCreateProgramWithSource(opt->context, 1, &ba_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating BA program: %d\n", err);
        goto cleanup;
    }
    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building BA program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        goto cleanup;
    }

    init_kernel = clCreateKernel(program, "initialize_fitness", &err);
    freq_kernel = clCreateKernel(program, "frequency_update", &err);
    local_kernel = clCreateKernel(program, "local_search", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating BA kernels: %d\n", err);
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

    // Query kernel work-group size
    size_t init_preferred_multiple, freq_preferred_multiple, local_preferred_multiple, best_preferred_multiple;
    err = clGetKernelWorkGroupInfo(init_kernel, opt->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                   sizeof(size_t), &init_preferred_multiple, NULL);
    err |= clGetKernelWorkGroupInfo(freq_kernel, opt->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                    sizeof(size_t), &freq_preferred_multiple, NULL);
    err |= clGetKernelWorkGroupInfo(local_kernel, opt->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                    sizeof(size_t), &local_preferred_multiple, NULL);
    err |= clGetKernelWorkGroupInfo(best_kernel, opt->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                    sizeof(size_t), &best_preferred_multiple, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Warning: Error querying kernel preferred work-group size: %d, using default %zu\n", err, local_work_size);
    } else {
        local_work_size = init_preferred_multiple < local_work_size ? init_preferred_multiple : local_work_size;
        printf("Preferred work-group size multiple: init=%zu, freq=%zu, local=%zu, best=%zu\n", 
               init_preferred_multiple, freq_preferred_multiple, local_preferred_multiple, best_preferred_multiple);
    }

    // Allocate GPU buffers
    freq_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(float), NULL, &err);
    velocities_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    best_pos_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    best_fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
    rng_states_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(uint), NULL, &err);
    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, dim * 2 * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(int), NULL, &err);
    opt->population_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    opt->fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating BA buffers: %d\n", err);
        goto cleanup;
    }

    // Allocate host memory
    positions = (float *)malloc(pop_size * dim * sizeof(float));
    fitness = (float *)malloc(pop_size * sizeof(float));
    best_pos = (float *)malloc(dim * sizeof(float));
    bounds = (float *)malloc(dim * 2 * sizeof(float));
    velocities = (float *)malloc(pop_size * dim * sizeof(float));
    freq = (float *)malloc(pop_size * sizeof(float));
    rng_states = (uint *)malloc(pop_size * sizeof(uint));
    best_idx_array = (int *)malloc(pop_size * sizeof(int));
    temp_position = (double *)malloc(dim * sizeof(double));
    if (!positions || !fitness || !best_pos || !bounds || !velocities || !freq || !rng_states || !best_idx_array || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }

    // Initialize host data
    float best_fitness = INFINITY;
    for (int d = 0; d < dim; d++) {
        best_pos[d] = (float)opt->best_solution.position[d];
        bounds[d * 2] = (float)opt->bounds[d * 2];     // Lower bound
        bounds[d * 2 + 1] = (float)opt->bounds[d * 2 + 1]; // Upper bound
    }
    for (int i = 0; i < pop_size; i++) {
        rng_states[i] = (uint)(time(NULL) ^ (i + 1));
        for (int d = 0; d < dim; d++) {
            positions[i * dim + d] = (float)opt->population[i].position[d];
            velocities[i * dim + d] = 0.0f;
        }
        freq[i] = 0.0f;
        fitness[i] = INFINITY; // Initialize fitness array
    }

    // Write initial data to GPU
    err = clEnqueueWriteBuffer(queue, opt->population_buffer, CL_TRUE, 0, 
                              pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, velocities_buffer, CL_TRUE, 0, 
                               pop_size * dim * sizeof(float), velocities, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, freq_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(float), freq, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, best_pos_buffer, CL_TRUE, 0, 
                               dim * sizeof(float), best_pos, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, best_fitness_buffer, CL_TRUE, 0, 
                               sizeof(float), &best_fitness, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, rng_states_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(uint), rng_states, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 
                               dim * 2 * sizeof(float), bounds, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(float), fitness, 0, NULL, NULL); // Initialize fitness buffer
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial buffers: %d\n", err);
        goto cleanup;
    }

    // Initialize fitness
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &opt->fitness_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(int), &pop_size);
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

    // Evaluate initial population fitness on CPU
    err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, 
                             pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading population buffer: %d\n", err);
        goto cleanup;
    }
    printf("Initial Population:\n");
    for (int i = 0; i < pop_size && i < 5; i++) {
        printf("Bat %d: [", i);
        for (int d = 0; d < dim; d++) {
            float p = positions[i * dim + d];
            float lower_bound = bounds[d * 2];
            float upper_bound = bounds[d * 2 + 1];
            if (p < lower_bound || p > upper_bound || !isfinite(p)) {
                fprintf(stderr, "Warning: Initial position out of bounds or non-finite for bat %d, dim %d: %f\n", i, d, p);
                p = lower_bound + (upper_bound - lower_bound) * ((float)rand() / RAND_MAX);
                positions[i * dim + d] = p;
            }
            temp_position[d] = (double)p;
            printf("%f", p);
            if (d < dim - 1) printf(", ");
        }
        printf("], Fitness: ");
        double fitness_value = objective_function(temp_position);
        if (!isfinite(fitness_value)) {
            fprintf(stderr, "Error: Non-finite initial fitness value for bat %d: %f\n", i, fitness_value);
            fitness_value = INFINITY;
        }
        printf("%f\n", fitness_value);
        opt->population[i].fitness = fitness_value;
        fitness[i] = (float)fitness_value;
        if (isfinite(fitness_value) && fitness_value < best_fitness) {
            best_fitness = (float)fitness_value;
            for (int d = 0; d < dim; d++) {
                best_pos[d] = positions[i * dim + d];
                opt->best_solution.position[d] = (double)best_pos[d];
            }
            opt->best_solution.fitness = fitness_value;
        }
    }
    if (isfinite(best_fitness)) {
        printf("Initial Best Position: [");
        for (int d = 0; d < dim; d++) {
            printf("%f", best_pos[d]);
            if (d < dim - 1) printf(", ");
        }
        printf("], Fitness: %f\n", best_fitness);
    } else {
        fprintf(stderr, "Error: No valid initial best fitness found\n");
        goto cleanup;
    }

    err = clEnqueueWriteBuffer(queue, opt->population_buffer, CL_TRUE, 0, 
                              pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(float), fitness, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, best_pos_buffer, CL_TRUE, 0, 
                               dim * sizeof(float), best_pos, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, best_fitness_buffer, CL_TRUE, 0, 
                               sizeof(float), &best_fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial buffers: %d\n", err);
        goto cleanup;
    }

    // Main optimization loop
    float loudness = LOUDNESS;
    float pulse_rate = PULSE_RATE;
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Update pulse rate and loudness
        pulse_rate = PULSE_RATE * (1.0f - exp(-GAMMA * iter));
        loudness *= ALPHA_BA;

        // Frequency update and velocity adjustment
        err = clSetKernelArg(freq_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(freq_kernel, 1, sizeof(cl_mem), &velocities_buffer);
        err |= clSetKernelArg(freq_kernel, 2, sizeof(cl_mem), &freq_buffer);
        err |= clSetKernelArg(freq_kernel, 3, sizeof(cl_mem), &best_pos_buffer);
        err |= clSetKernelArg(freq_kernel, 4, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(freq_kernel, 5, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(freq_kernel, 6, sizeof(int), &dim);
        err |= clSetKernelArg(freq_kernel, 7, sizeof(int), &pop_size);
        err |= clSetKernelArg(freq_kernel, 8, sizeof(int), &iter);
        err |= clSetKernelArg(freq_kernel, 9, sizeof(int), &max_iter);
        err |= clSetKernelArg(freq_kernel, 10, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting frequency kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, freq_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing frequency kernel: %d\n", err);
            goto cleanup;
        }

        // Local search
        err = clSetKernelArg(local_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(local_kernel, 1, sizeof(cl_mem), &best_pos_buffer);
        err |= clSetKernelArg(local_kernel, 2, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(local_kernel, 3, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(local_kernel, 4, sizeof(float), &pulse_rate);
        err |= clSetKernelArg(local_kernel, 5, sizeof(float), &loudness);
        err |= clSetKernelArg(local_kernel, 6, sizeof(int), &dim);
        err |= clSetKernelArg(local_kernel, 7, sizeof(int), &pop_size);
        err |= clSetKernelArg(local_kernel, 8, sizeof(int), &iter);
        err |= clSetKernelArg(local_kernel, 9, sizeof(int), &max_iter);
        err |= clSetKernelArg(local_kernel, 10, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting local search kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, local_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing local search kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate fitness on CPU with precision handling
        err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, 
                                 pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(queue, velocities_buffer, CL_TRUE, 0, 
                                  pop_size * dim * sizeof(float), velocities, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(queue, freq_buffer, CL_TRUE, 0, 
                                  pop_size * sizeof(float), freq, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading population/velocities/freq buffers: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < pop_size; i++) {
            // Validate and log velocity magnitude
            float vel_mag = 0.0f;
            for (int d = 0; d < dim; d++) {
                vel_mag += velocities[i * dim + d] * velocities[i * dim + d];
            }
            vel_mag = sqrt(vel_mag);
            if (iter % 100 == 0 && i < 5) {
                printf("Bat %d at iteration %d: Velocity Magnitude: %f\n", i, iter + 1, vel_mag);
            }
            // Validate positions
            for (int d = 0; d < dim; d++) {
                float p = positions[i * dim + d];
                float lower_bound = bounds[d * 2];
                float upper_bound = bounds[d * 2 + 1];
                if (!isfinite(p)) {
                    fprintf(stderr, "Error: Non-finite position for bat %d, dim %d at iteration %d: %f\n", 
                            i, d, iter + 1, p);
                    p = lower_bound + (upper_bound - lower_bound) * ((float)rand() / RAND_MAX);
                    positions[i * dim + d] = p;
                } else if (p < lower_bound || p > upper_bound) {
                    fprintf(stderr, "Warning: Position out of bounds for bat %d, dim %d at iteration %d: %f\n", 
                            i, d, iter + 1, p);
                    p = lower_bound + (upper_bound - lower_bound) * ((float)rand() / RAND_MAX);
                    positions[i * dim + d] = p;
                }
                temp_position[d] = (double)p;
            }
            // Compute fitness
            double new_fitness = objective_function(temp_position);
            if (!isfinite(new_fitness)) {
                fprintf(stderr, "Error: Non-finite fitness value for bat %d at iteration %d: %f, Position: [", 
                        i, iter + 1, new_fitness);
                for (int d = 0; d < dim; d++) {
                    fprintf(stderr, "%f", temp_position[d]);
                    if (d < dim - 1) fprintf(stderr, ", ");
                }
                fprintf(stderr, "]\n");
                new_fitness = INFINITY;
            }
            // Log for debugging
            if (iter % 100 == 0 && i < 5) {
                printf("Bat %d at iteration %d: Position: [", i, iter + 1);
                for (int d = 0; d < dim; d++) {
                    printf("%f", temp_position[d]);
                    if (d < dim - 1) printf(", ");
                }
                printf("], Fitness: %f\n", new_fitness);
            }
            // Accept better solutions
            if (isfinite(new_fitness) && new_fitness <= opt->population[i].fitness && 
                ((float)rand() / RAND_MAX) > loudness) {
                opt->population[i].fitness = new_fitness;
                fitness[i] = (float)new_fitness;
                // Update position in GPU buffer
                for (int d = 0; d < dim; d++) {
                    positions[i * dim + d] = (float)temp_position[d];
                }
            }
            if (isfinite(new_fitness) && new_fitness < best_fitness) {
                best_fitness = (float)new_fitness;
                for (int d = 0; d < dim; d++) {
                    best_pos[d] = positions[i * dim + d];
                    opt->best_solution.position[d] = temp_position[d];
                }
                opt->best_solution.fitness = new_fitness;
                if (iter % 100 == 0) {
                    printf("Updated best fitness at bat %d, iteration %d: %f\n", i, iter + 1, new_fitness);
                }
            }
        }

        // Update velocities with double precision on CPU
        for (int i = 0; i < pop_size; i++) {
            for (int d = 0; d < dim; d++) {
                double temp_vel = (double)velocities[i * dim + d] + 
                                 ((double)positions[i * dim + d] - (double)best_pos[d]) * (double)freq[i];
                float lower_bound = bounds[d * 2];
                float upper_bound = bounds[d * 2 + 1];
                float domain_size = upper_bound - lower_bound;
                velocities[i * dim + d] = (float)clamp((float)temp_vel, -0.1f * domain_size, 0.1f * domain_size);
            }
        }

        err = clEnqueueWriteBuffer(queue, opt->population_buffer, CL_TRUE, 0, 
                                  pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, velocities_buffer, CL_TRUE, 0, 
                                   pop_size * dim * sizeof(float), velocities, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, freq_buffer, CL_TRUE, 0, 
                                   pop_size * sizeof(float), freq, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, 
                                   pop_size * sizeof(float), fitness, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, best_pos_buffer, CL_TRUE, 0, 
                                   dim * sizeof(float), best_pos, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, best_fitness_buffer, CL_TRUE, 0, 
                                   sizeof(float), &best_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing buffers: %d\n", err);
            goto cleanup;
        }

        // Find best solution
        err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &best_idx_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(int), &pop_size);
        err |= clSetKernelArg(best_kernel, 3, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 4, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[3]);
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
        if (best_idx < 0 || best_idx >= pop_size) {
            fprintf(stderr, "Error: Invalid best_idx %d at iteration %d\n", best_idx, iter + 1);
            continue; // Skip invalid index
        }
        float best_fitness_check;
        err = clEnqueueReadBuffer(queue, opt->fitness_buffer, CL_TRUE, best_idx * sizeof(float), 
                                 sizeof(float), &best_fitness_check, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best fitness: %d\n", err);
            goto cleanup;
        }
        if (iter % 100 == 0) {
            printf("Best idx %d at iteration %d, fitness from buffer: %f\n", best_idx, iter + 1, best_fitness_check);
        }
        if (isfinite(best_fitness_check) && best_fitness_check < best_fitness && best_fitness_check > 0.0f) {
            // Verify fitness with objective_function
            float *best_solution = (float *)malloc(dim * sizeof(float));
            if (!best_solution) {
                fprintf(stderr, "Error: Memory allocation failed for best_solution\n");
                goto cleanup;
            }
            err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, best_idx * dim * sizeof(float), 
                                     dim * sizeof(float), best_solution, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best solution: %d\n", err);
                free(best_solution);
                goto cleanup;
            }
            for (int d = 0; d < dim; d++) {
                temp_position[d] = (double)best_solution[d];
            }
            double verified_fitness = objective_function(temp_position);
            if (isfinite(verified_fitness) && verified_fitness < best_fitness) {
                best_fitness = (float)verified_fitness;
                for (int d = 0; d < dim; d++) {
                    opt->best_solution.position[d] = (double)best_solution[d];
                    best_pos[d] = best_solution[d];
                }
                opt->best_solution.fitness = verified_fitness;
                if (iter % 100 == 0) {
                    printf("Verified best fitness at idx %d, iteration %d: %f, Position: [", 
                           best_idx, iter + 1, verified_fitness);
                    for (int d = 0; d < dim; d++) {
                        printf("%f", best_solution[d]);
                        if (d < dim - 1) printf(", ");
                    }
                    printf("]\n");
                }
            } else {
                fprintf(stderr, "Warning: Invalid verified fitness %f for best_idx %d at iteration %d\n", 
                        verified_fitness, best_idx, iter + 1);
            }
            free(best_solution);
            err = clEnqueueWriteBuffer(queue, best_pos_buffer, CL_TRUE, 0, 
                                      dim * sizeof(float), best_pos, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(queue, best_fitness_buffer, CL_TRUE, 0, 
                                       sizeof(float), &best_fitness, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing best buffers: %d\n", err);
                goto cleanup;
            }
        }

        // Profiling
        err = clFinish(queue);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error finishing queue: %d\n", err);
            goto cleanup;
        }
        cl_ulong time_start, time_end;
        double init_time = 0.0, freq_time = 0.0, local_time = 0.0, best_time = 0.0;
        for (int i = 0; i < 4; i++) {
            if (events[i]) {
                err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                err |= clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                if (err == CL_SUCCESS) {
                    double time_ms = (time_end - time_start) / 1e6;
                    if (i == 0) init_time = time_ms;
                    else if (i == 1) freq_time = time_ms;
                    else if (i == 2) local_time = time_ms;
                    else if (i == 3) best_time = time_ms;
                }
            }
        }
        end_time = get_time_ms();
        printf("BA|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Frequency: %.3f ms | Local Search: %.3f ms | Best: %.3f ms\n",
               iter + 1, opt->best_solution.fitness, end_time - start_time, 
               init_time, freq_time, local_time, best_time);

        // Debugging output every 100 iterations
        if (iter % 100 == 0) {
            printf("Iteration %d: Best Fitness = %f, Best Position: [", iter + 1, opt->best_solution.fitness);
            for (int d = 0; d < dim; d++) {
                printf("%f", opt->best_solution.position[d]);
                if (d < dim - 1) printf(", ");
            }
            printf("]\n");
        }

        // Reset events
        for (int i = 0; i < 4; i++) {
            if (events[i]) {
                clReleaseEvent(events[i]);
                events[i] = 0;
            }
        }
    }

    // Update CPU-side population
    err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, 
                             pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final population buffer: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < pop_size; i++) {
        for (int d = 0; d < dim; d++) {
            float p = positions[i * dim + d];
            float lower_bound = bounds[d * 2];
            float upper_bound = bounds[d * 2 + 1];
            if (p < lower_bound || p > upper_bound || !isfinite(p)) {
                fprintf(stderr, "Warning: Final position out of bounds or non-finite for bat %d, dim %d: %f\n", i, d, p);
                p = lower_bound + (upper_bound - lower_bound) * ((float)rand() / RAND_MAX);
                positions[i * dim + d] = p;
            }
            opt->population[i].position[d] = (double)p;
        }
        for (int d = 0; d < dim; d++) {
            temp_position[d] = opt->population[i].position[d];
        }
        opt->population[i].fitness = objective_function(temp_position);
    }

cleanup:
    if (positions) free(positions);
    if (fitness) free(fitness);
    if (best_pos) free(best_pos);
    if (bounds) free(bounds);
    if (velocities) free(velocities);
    if (freq) free(freq);
    if (rng_states) free(rng_states);
    if (best_idx_array) free(best_idx_array);
    if (temp_position) free(temp_position);
    if (freq_buffer) clReleaseMemObject(freq_buffer);
    if (velocities_buffer) clReleaseMemObject(velocities_buffer);
    if (best_pos_buffer) clReleaseMemObject(best_pos_buffer);
    if (best_fitness_buffer) clReleaseMemObject(best_fitness_buffer);
    if (rng_states_buffer) clReleaseMemObject(rng_states_buffer);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (opt->population_buffer) clReleaseMemObject(opt->population_buffer);
    if (opt->fitness_buffer) clReleaseMemObject(opt->fitness_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (freq_kernel) clReleaseKernel(freq_kernel);
    if (local_kernel) clReleaseKernel(local_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 4; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) {
        clFinish(queue);
        clReleaseCommandQueue(queue);
    }
}
