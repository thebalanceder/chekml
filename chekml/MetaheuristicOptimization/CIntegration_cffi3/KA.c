#include "KA.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
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

// OpenCL kernel source
static const char* ka_kernel_source =
"#define P1 0.2f\n"
"#define P2 0.5f\n"
"#define S_MAX 4\n"
// LCG random number generator
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
// Initialize population
"__kernel void initialize_population(\n"
"    __global float* population,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < pop_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float min = local_bounds[2 * d];\n"
"            float max = local_bounds[2 * d + 1];\n"
"            population[id * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"        }\n"
"    }\n"
"}\n"
// Find nearest neighbor
"__kernel void find_nearest_neighbor(\n"
"    __global const float* population,\n"
"    __global int* nn_indices,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    const int m1)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < m1) {\n"
"        float min_dist = 1.0e30f;\n"
"        int nearest_idx = -1;\n"
"        for (int i = 0; i < pop_size; i++) {\n"
"            if (i == id) continue;\n"
"            float dist = 0.0f;\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float diff = population[i * dim + d] - population[id * dim + d];\n"
"                dist += diff * diff;\n"
"            }\n"
"            if (dist < min_dist) {\n"
"                min_dist = dist;\n"
"                nearest_idx = i;\n"
"            }\n"
"        }\n"
"        nn_indices[id] = nearest_idx;\n"
"    }\n"
"}\n"
// Swirl motion for best group
"__kernel void swirl_motion(\n"
"    __global float* population,\n"
"    __global float* trial,\n"
"    __global const float* bounds,\n"
"    __global const int* nn_indices,\n"
"    const int dim,\n"
"    const int m1,\n"
"    const int s,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < m1) {\n"
"        uint local_seed = seed + id * dim;\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        float swirl_strength = (float)(S_MAX - s + 1) / S_MAX;\n"
"        int nn_idx = nn_indices[id];\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float nn_pos = population[nn_idx * dim + d];\n"
"            float rand_val = lcg_rand_float(&local_seed, -1.0f, 1.0f);\n"
"            trial[id * dim + d] = population[id * dim + d] + swirl_strength * (nn_pos - population[id * dim + d]) * rand_val;\n"
"            float min = local_bounds[2 * d];\n"
"            float max = local_bounds[2 * d + 1];\n"
"            trial[id * dim + d] = clamp(trial[id * dim + d], min, max);\n"
"        }\n"
"    }\n"
"}\n"
// Crossover for middle group
"__kernel void crossover_middle(\n"
"    __global float* population,\n"
"    __global float* trial,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int m1,\n"
"    const int m2,\n"
"    const int pop_size,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < m2) {\n"
"        uint local_seed = seed + id * dim;\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        int idx = id + m1;\n"
"        int a = lcg_rand(&local_seed) % pop_size;\n"
"        int b = lcg_rand(&local_seed) % pop_size;\n"
"        while (a == idx) a = lcg_rand(&local_seed) % pop_size;\n"
"        while (b == idx || b == a) b = lcg_rand(&local_seed) % pop_size;\n"
"        float w1 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float w2 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float w3 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float sum = w1 + w2 + w3;\n"
"        w1 /= sum; w2 /= sum; w3 /= sum;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            trial[idx * dim + d] = w1 * population[idx * dim + d] + w2 * population[a * dim + d] + w3 * population[b * dim + d];\n"
"            float min = local_bounds[2 * d];\n"
"            float max = local_bounds[2 * d + 1];\n"
"            trial[idx * dim + d] = clamp(trial[idx * dim + d], min, max);\n"
"        }\n"
"    }\n"
"}\n"
// Reinitialize worst group
"__kernel void reinitialize_worst(\n"
"    __global float* population,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int m1,\n"
"    const int m2,\n"
"    const int m3,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < m3) {\n"
"        uint local_seed = seed + id * dim;\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        int idx = id + m1 + m2;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float min = local_bounds[2 * d];\n"
"            float max = local_bounds[2 * d + 1];\n"
"            population[idx * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"        }\n"
"    }\n"
"}\n"
// Find best solution
"__kernel void find_best_solution(\n"
"    __global const float* fitness,\n"
"    __global const float* population,\n"
"    __global float* best_position,\n"
"    __global float* best_fitness,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    __local float* local_fitness,\n"
"    __local int* local_indices)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    if (id < pop_size) {\n"
"        local_fitness[local_id] = fitness[id];\n"
"        local_indices[local_id] = id;\n"
"    } else {\n"
"        local_fitness[local_id] = 1.0e30f;\n"
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
"            *best_fitness = local_fitness[0];\n"
"            for (int d = 0; d < dim; d++) {\n"
"                best_position[d] = population[idx * dim + d];\n"
"            }\n"
"        }\n"
"    }\n"
"}\n";

void KA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, nn_kernel = NULL, swirl_kernel = NULL, crossover_kernel = NULL, reinitialize_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, population_buffer = NULL, trial_buffer = NULL, fitness_buffer = NULL, nn_indices_buffer = NULL, best_position_buffer = NULL, best_fitness_buffer = NULL;
    float *bounds_float = NULL, *population = NULL, *trial = NULL, *fitness = NULL, *best_position = NULL, *best_fitness = NULL;
    int *nn_indices = NULL;
    double *temp_position = NULL;
    cl_event events[4] = {0};
    double start_time, end_time;

    // Validate input
    if (!opt || !opt->population || !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: Population[%d].position is null\n", i);
            exit(EXIT_FAILURE);
        }
    }

    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    const int max_iter = opt->max_iter;
    const int m1 = (int)(KA_P1 * pop_size);
    const int m2 = 2 * ((int)(KA_P2 * pop_size) / 2);
    const int m3 = pop_size - m1 - m2;
    if (dim < 1 || pop_size < 3 || max_iter < 1 || m1 < 0 || m2 < 0 || m3 < 0) {
        fprintf(stderr, "Error: Invalid dim (%d), pop_size (%d), max_iter (%d), m1 (%d), m2 (%d), or m3 (%d)\n",
                dim, pop_size, max_iter, m1, m2, m3);
        exit(EXIT_FAILURE);
    }

    // Select GPU platform and device
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "Error: No OpenCL platforms found: %d\n", err);
        exit(EXIT_FAILURE);
    }
    cl_platform_id *platforms = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) {
        fprintf(stderr, "Error: Memory allocation failed for platforms\n");
        exit(EXIT_FAILURE);
    }
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error getting platform IDs: %d\n", err);
        free(platforms);
        exit(EXIT_FAILURE);
    }

    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) continue;
        cl_device_id *devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
        if (!devices) continue;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
        if (err != CL_SUCCESS) {
            free(devices);
            continue;
        }
        platform = platforms[i];
        device = devices[0];
        free(devices);
        break;
    }
    free(platforms);
    if (!platform || !device) {
        fprintf(stderr, "Error: No GPU device found\n");
        exit(EXIT_FAILURE);
    }

    // Create context and queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating OpenCL context: %d\n", err);
        exit(EXIT_FAILURE);
    }
    cl_queue_properties queue_props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, queue_props, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating OpenCL queue: %d\n", err);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }

    // Create program
    program = clCreateProgramWithSource(context, 1, &ka_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating KA program: %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building KA program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    nn_kernel = clCreateKernel(program, "find_nearest_neighbor", &err);
    swirl_kernel = clCreateKernel(program, "swirl_motion", &err);
    crossover_kernel = clCreateKernel(program, "crossover_middle", &err);
    reinitialize_kernel = clCreateKernel(program, "reinitialize_worst", &err);
    best_kernel = clCreateKernel(program, "find_best_solution", &err);
    if (err != CL_SUCCESS || !init_kernel || !nn_kernel || !swirl_kernel || !crossover_kernel || !reinitialize_kernel || !best_kernel) {
        fprintf(stderr, "Error creating KA kernels: %d\n", err);
        goto cleanup;
    }

    // Query work-group size
    size_t max_work_group_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    size_t local_work_size = max_work_group_size < KA_LOCAL_WORK_SIZE ? max_work_group_size : KA_LOCAL_WORK_SIZE;

    // Create buffers
    bounds_float = (float *)malloc(2 * dim * sizeof(float));
    population = (float *)malloc(pop_size * dim * sizeof(float));
    trial = (float *)malloc(pop_size * dim * sizeof(float));
    fitness = (float *)malloc(pop_size * sizeof(float));
    nn_indices = (int *)malloc(m1 * sizeof(int));
    best_position = (float *)malloc(dim * sizeof(float));
    best_fitness = (float *)malloc(sizeof(float));
    temp_position = (double *)malloc(dim * sizeof(double));
    if (!bounds_float || !population || !trial || !fitness || !nn_indices || !best_position || !best_fitness || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < dim; i++) {
        bounds_float[2 * i] = (float)opt->bounds[i];
        bounds_float[2 * i + 1] = (float)opt->bounds[dim + i];
        best_position[i] = 0.0f;
    }
    *best_fitness = 1.0e30f;

    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    trial_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * sizeof(float), NULL, &err);
    nn_indices_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, m1 * sizeof(int), NULL, &err);
    best_position_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    best_fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !population_buffer || !trial_buffer || !fitness_buffer || !nn_indices_buffer || !best_position_buffer || !best_fitness_buffer) {
        fprintf(stderr, "Error creating KA buffers: %d\n", err);
        goto cleanup;
    }

    // Write bounds and initial best solution
    err = clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, best_position_buffer, CL_TRUE, 0, dim * sizeof(float), best_position, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, best_fitness_buffer, CL_TRUE, 0, sizeof(float), best_fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial buffers: %d\n", err);
        goto cleanup;
    }

    // Initialize best solution
    opt->best_solution.fitness = 1.0e30;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = 0.0;
    }

    // Initialize population
    uint seed = (uint)time(NULL);
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 3, sizeof(int), &pop_size);
    err |= clSetKernelArg(init_kernel, 4, sizeof(uint), &seed);
    err |= clSetKernelArg(init_kernel, 5, 2 * dim * sizeof(float), NULL);
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

    // Evaluate initial population
    err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, 0, pop_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading population buffer: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < pop_size; i++) {
        for (int d = 0; d < dim; d++) {
            temp_position[d] = (double)population[i * dim + d];
        }
        fitness[i] = (float)objective_function(temp_position);
        if (!isfinite(fitness[i])) {
            fprintf(stderr, "Warning: Invalid fitness value at index %d\n", i);
            fitness[i] = 1.0e30f;
        }
        if (fitness[i] < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)fitness[i];
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)population[i * dim + d];
            }
        }
    }
    err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, pop_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing fitness buffer: %d\n", err);
        goto cleanup;
    }

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // m1: Best group with swirl
        if (m1 > 0) {
            err = clSetKernelArg(nn_kernel, 0, sizeof(cl_mem), &population_buffer);
            err |= clSetKernelArg(nn_kernel, 1, sizeof(cl_mem), &nn_indices_buffer);
            err |= clSetKernelArg(nn_kernel, 2, sizeof(int), &dim);
            err |= clSetKernelArg(nn_kernel, 3, sizeof(int), &pop_size);
            err |= clSetKernelArg(nn_kernel, 4, sizeof(int), &m1);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error setting nn kernel args: %d\n", err);
                goto cleanup;
            }
            size_t nn_global_size = ((m1 + local_work_size - 1) / local_work_size) * local_work_size;
            err = clEnqueueNDRangeKernel(queue, nn_kernel, 1, NULL, &nn_global_size, &local_work_size, 0, NULL, &events[0]);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error enqueuing nn kernel: %d\n", err);
                goto cleanup;
            }

            for (int S = 1; S <= 2 * KA_S_MAX - 1; S++) {
                err = clSetKernelArg(swirl_kernel, 0, sizeof(cl_mem), &population_buffer);
                err |= clSetKernelArg(swirl_kernel, 1, sizeof(cl_mem), &trial_buffer);
                err |= clSetKernelArg(swirl_kernel, 2, sizeof(cl_mem), &bounds_buffer);
                err |= clSetKernelArg(swirl_kernel, 3, sizeof(cl_mem), &nn_indices_buffer);
                err |= clSetKernelArg(swirl_kernel, 4, sizeof(int), &dim);
                err |= clSetKernelArg(swirl_kernel, 5, sizeof(int), &m1);
                err |= clSetKernelArg(swirl_kernel, 6, sizeof(int), &S);
                err |= clSetKernelArg(swirl_kernel, 7, sizeof(uint), &seed);
                err |= clSetKernelArg(swirl_kernel, 8, 2 * dim * sizeof(float), NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error setting swirl kernel args: %d\n", err);
                    goto cleanup;
                }
                err = clEnqueueNDRangeKernel(queue, swirl_kernel, 1, NULL, &nn_global_size, &local_work_size, 0, NULL, &events[1]);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error enqueuing swirl kernel: %d\n", err);
                    goto cleanup;
                }

                // Evaluate trial solutions
                err = clEnqueueReadBuffer(queue, trial_buffer, CL_TRUE, 0, m1 * dim * sizeof(float), trial, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error reading trial buffer: %d\n", err);
                    goto cleanup;
                }
                for (int i = 0; i < m1; i++) {
                    for (int d = 0; d < dim; d++) {
                        temp_position[d] = (double)trial[i * dim + d];
                    }
                    float trial_fitness = (float)objective_function(temp_position);
                    if (!isfinite(trial_fitness)) {
                        trial_fitness = 1.0e30f;
                    }
                    if (trial_fitness < fitness[i]) {
                        fitness[i] = trial_fitness;
                        for (int d = 0; d < dim; d++) {
                            population[i * dim + d] = trial[i * dim + d];
                        }
                        if (trial_fitness < opt->best_solution.fitness) {
                            opt->best_solution.fitness = (double)trial_fitness;
                            for (int d = 0; d < dim; d++) {
                                opt->best_solution.position[d] = (double)trial[i * dim + d];
                            }
                        }
                        // Update nearest neighbor
                        err = clEnqueueWriteBuffer(queue, population_buffer, CL_TRUE, i * dim * sizeof(float), dim * sizeof(float), &population[i * dim], 0, NULL, NULL);
                        if (err != CL_SUCCESS) {
                            fprintf(stderr, "Error writing updated population: %d\n", err);
                            goto cleanup;
                        }
                        err = clEnqueueNDRangeKernel(queue, nn_kernel, 1, NULL, &nn_global_size, &local_work_size, 0, NULL, NULL);
                        if (err != CL_SUCCESS) {
                            fprintf(stderr, "Error enqueuing nn kernel after swirl: %d\n", err);
                            goto cleanup;
                        }
                        err = clEnqueueReadBuffer(queue, nn_indices_buffer, CL_TRUE, i * sizeof(int), sizeof(int), &nn_indices[i], 0, NULL, NULL);
                        if (err != CL_SUCCESS) {
                            fprintf(stderr, "Error reading nn_indices: %d\n", err);
                            goto cleanup;
                        }
                    }
                }
                err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, m1 * sizeof(float), fitness, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error writing fitness after swirl: %d\n", err);
                    goto cleanup;
                }
            }
        }

        // m2: Middle group with crossover
        if (m2 > 0) {
            err = clSetKernelArg(crossover_kernel, 0, sizeof(cl_mem), &population_buffer);
            err |= clSetKernelArg(crossover_kernel, 1, sizeof(cl_mem), &trial_buffer);
            err |= clSetKernelArg(crossover_kernel, 2, sizeof(cl_mem), &bounds_buffer);
            err |= clSetKernelArg(crossover_kernel, 3, sizeof(int), &dim);
            err |= clSetKernelArg(crossover_kernel, 4, sizeof(int), &m1);
            err |= clSetKernelArg(crossover_kernel, 5, sizeof(int), &m2);
            err |= clSetKernelArg(crossover_kernel, 6, sizeof(int), &pop_size);
            err |= clSetKernelArg(crossover_kernel, 7, sizeof(uint), &seed);
            err |= clSetKernelArg(crossover_kernel, 8, 2 * dim * sizeof(float), NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error setting crossover kernel args: %d\n", err);
                goto cleanup;
            }
            size_t crossover_global_size = ((m2 + local_work_size - 1) / local_work_size) * local_work_size;
            err = clEnqueueNDRangeKernel(queue, crossover_kernel, 1, NULL, &crossover_global_size, &local_work_size, 0, NULL, &events[2]);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error enqueuing crossover kernel: %d\n", err);
                goto cleanup;
            }

            // Evaluate trial solutions
            err = clEnqueueReadBuffer(queue, trial_buffer, CL_TRUE, m1 * dim * sizeof(float), m2 * dim * sizeof(float), &trial[m1 * dim], 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading trial buffer for crossover: %d\n", err);
                goto cleanup;
            }
            for (int i = 0; i < m2; i++) {
                int idx = i + m1;
                for (int d = 0; d < dim; d++) {
                    temp_position[d] = (double)trial[idx * dim + d];
                }
                float trial_fitness = (float)objective_function(temp_position);
                if (!isfinite(trial_fitness)) {
                    trial_fitness = 1.0e30f;
                }
                if (trial_fitness < fitness[idx]) {
                    fitness[idx] = trial_fitness;
                    for (int d = 0; d < dim; d++) {
                        population[idx * dim + d] = trial[idx * dim + d];
                    }
                    if (trial_fitness < opt->best_solution.fitness) {
                        opt->best_solution.fitness = (double)trial_fitness;
                        for (int d = 0; d < dim; d++) {
                            opt->best_solution.position[d] = (double)trial[idx * dim + d];
                        }
                    }
                }
            }
            err = clEnqueueWriteBuffer(queue, population_buffer, CL_TRUE, m1 * dim * sizeof(float), m2 * dim * sizeof(float), &population[m1 * dim], 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, m1 * sizeof(float), m2 * sizeof(float), &fitness[m1], 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing population/fitness after crossover: %d\n", err);
                goto cleanup;
            }
        }

        // m3: Reinitialize worst group
        if (m3 > 0) {
            err = clSetKernelArg(reinitialize_kernel, 0, sizeof(cl_mem), &population_buffer);
            err |= clSetKernelArg(reinitialize_kernel, 1, sizeof(cl_mem), &bounds_buffer);
            err |= clSetKernelArg(reinitialize_kernel, 2, sizeof(int), &dim);
            err |= clSetKernelArg(reinitialize_kernel, 3, sizeof(int), &m1);
            err |= clSetKernelArg(reinitialize_kernel, 4, sizeof(int), &m2);
            err |= clSetKernelArg(reinitialize_kernel, 5, sizeof(int), &m3);
            err |= clSetKernelArg(reinitialize_kernel, 6, sizeof(uint), &seed);
            err |= clSetKernelArg(reinitialize_kernel, 7, 2 * dim * sizeof(float), NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error setting reinitialize kernel args: %d\n", err);
                goto cleanup;
            }
            size_t reinitialize_global_size = ((m3 + local_work_size - 1) / local_work_size) * local_work_size;
            err = clEnqueueNDRangeKernel(queue, reinitialize_kernel, 1, NULL, &reinitialize_global_size, &local_work_size, 0, NULL, &events[3]);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error enqueuing reinitialize kernel: %d\n", err);
                goto cleanup;
            }

            // Evaluate reinitialized solutions
            err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, (m1 + m2) * dim * sizeof(float), m3 * dim * sizeof(float), &population[(m1 + m2) * dim], 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading population for reinitialize: %d\n", err);
                goto cleanup;
            }
            for (int i = 0; i < m3; i++) {
                int idx = i + m1 + m2;
                for (int d = 0; d < dim; d++) {
                    temp_position[d] = (double)population[idx * dim + d];
                }
                fitness[idx] = (float)objective_function(temp_position);
                if (!isfinite(fitness[idx])) {
                    fitness[idx] = 1.0e30f;
                }
                if (fitness[idx] < opt->best_solution.fitness) {
                    opt->best_solution.fitness = (double)fitness[idx];
                    for (int d = 0; d < dim; d++) {
                        opt->best_solution.position[d] = (double)population[idx * dim + d];
                    }
                }
            }
            err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, (m1 + m2) * sizeof(float), m3 * sizeof(float), &fitness[m1 + m2], 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing fitness after reinitialize: %d\n", err);
                goto cleanup;
            }
        }

        // Find best solution
        err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(cl_mem), &best_position_buffer);
        err |= clSetKernelArg(best_kernel, 3, sizeof(cl_mem), &best_fitness_buffer);
        err |= clSetKernelArg(best_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(best_kernel, 5, sizeof(int), &pop_size);
        err |= clSetKernelArg(best_kernel, 6, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 7, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best_solution kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing best_solution kernel: %d\n", err);
            goto cleanup;
        }

        // Update best solution
        err = clEnqueueReadBuffer(queue, best_fitness_buffer, CL_TRUE, 0, sizeof(float), best_fitness, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(queue, best_position_buffer, CL_TRUE, 0, dim * sizeof(float), best_position, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best solution/fitness: %d\n", err);
            goto cleanup;
        }
        if (*best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)*best_fitness;
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)best_position[d];
            }
        }

        // Enforce boundary constraints
        enforce_bound_constraints(opt);

        // Profiling with best solution position
        cl_ulong time_start, time_end;
        double nn_time = 0, swirl_time = 0, crossover_time = 0, reinitialize_time = 0;
        if (events[0]) {
            clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            nn_time = (time_end - time_start) / 1e6;
        }
        if (events[1]) {
            clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            swirl_time = (time_end - time_start) / 1e6;
        }
        if (events[2]) {
            clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            crossover_time = (time_end - time_start) / 1e6;
        }
        if (events[3]) {
            clGetEventProfilingInfo(events[3], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[3], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            reinitialize_time = (time_end - time_start) / 1e6;
        }
        end_time = get_time_ms();
        printf("KA|Iter %4d -----> Fitness: %9.5f | Pos: [", iter + 1, opt->best_solution.fitness);
        for (int d = 0; d < dim && d < 3; d++) {
            printf("%.3f%s", opt->best_solution.position[d], d < dim - 1 && d < 2 ? ", " : "");
        }
        if (dim > 3) printf("...");
        printf("] | Total: %.3f ms | NN: %.3f ms | Swirl: %.3f ms | Crossover: %.3f ms | Reinit: %.3f ms\n",
               end_time - start_time, nn_time, swirl_time, crossover_time, reinitialize_time);
    }

    // Update CPU-side population
    err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, 0, pop_size * dim * sizeof(float), population, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, 0, pop_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final population/fitness: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < pop_size; i++) {
        for (int d = 0; d < dim; d++) {
            opt->population[i].position[d] = (double)population[i * dim + d];
        }
        opt->population[i].fitness = (double)fitness[i];
    }

cleanup:
    if (bounds_float) free(bounds_float);
    if (population) free(population);
    if (trial) free(trial);
    if (fitness) free(fitness);
    if (nn_indices) free(nn_indices);
    if (best_position) free(best_position);
    if (best_fitness) free(best_fitness);
    if (temp_position) free(temp_position);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (population_buffer) clReleaseMemObject(population_buffer);
    if (trial_buffer) clReleaseMemObject(trial_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (nn_indices_buffer) clReleaseMemObject(nn_indices_buffer);
    if (best_position_buffer) clReleaseMemObject(best_position_buffer);
    if (best_fitness_buffer) clReleaseMemObject(best_fitness_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (nn_kernel) clReleaseKernel(nn_kernel);
    if (swirl_kernel) clReleaseKernel(swirl_kernel);
    if (crossover_kernel) clReleaseKernel(crossover_kernel);
    if (reinitialize_kernel) clReleaseKernel(reinitialize_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 4; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (queue) clFinish(queue);
}
