/* GA.c - GPU-Optimized Genetic Algorithm for Extreme Speed */
#include "GA.h"
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

// Debug logging macro
#if GA_DEBUG
#define DEBUG_LOG(fmt, ...) fprintf(stderr, "GA DEBUG: " fmt, ##__VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif

// OpenCL kernel source
static const char* ga_kernel_source =
"// Simple LCG random number generator\n"
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
// Enforce bound constraints
"void enforce_bounds(__global float* position, __global const float* bounds, int dim) {\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float min = bounds[2 * d];\n"
"        float max = bounds[2 * d + 1];\n"
"        position[d] = fmax(min, fmin(max, position[d]));\n"
"    }\n"
"}\n"
// Initialize Population Kernel
"__kernel void initialize_population(\n"
"    __global float* population,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < pop_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            population[id * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"        }\n"
"        enforce_bounds(&population[id * dim], bounds, dim);\n"
"    }\n"
"}\n"
// Crossover Kernel
"__kernel void crossover(\n"
"    __global float* population,\n"
"    __global const float* best,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < pop_size && id != 0) { // Skip best individual\n"
"        uint local_seed = seed + id * dim;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            if (lcg_rand(&local_seed) % 2) {\n"
"                int parent = lcg_rand(&local_seed) % 2 ? 0 : 1; // Best or second-best\n"
"                population[id * dim + d] = parent == 0 ? best[d] : population[1 * dim + d];\n"
"            }\n"
"        }\n"
"        enforce_bounds(&population[id * dim], bounds, dim);\n"
"    }\n"
"}\n"
// Mutate Worst Kernel
"__kernel void mutate_worst(\n"
"    __global float* population,\n"
"    __global const float* bounds,\n"
"    __global const int* worst_idx,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id == 0) {\n"
"        int worst = worst_idx[0];\n"
"        if (worst >= 0 && worst < pop_size) {\n"
"            uint local_seed = seed + worst * dim;\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float min = bounds[2 * d];\n"
"                float max = bounds[2 * d + 1];\n"
"                population[worst * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"            }\n"
"            enforce_bounds(&population[worst * dim], bounds, dim);\n"
"        }\n"
"    }\n"
"}\n"
// Find Best and Worst Kernel
"__kernel void find_best_worst(\n"
"    __global const float* fitness,\n"
"    __global int* best_idx,\n"
"    __global int* worst_idx,\n"
"    const int pop_size,\n"
"    __local float* local_fitness,\n"
"    __local int* local_indices)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    float fitness_val = id < pop_size ? fitness[id] : INFINITY;\n"
"    local_fitness[local_id] = fitness_val;\n"
"    local_indices[local_id] = id < pop_size ? id : -1;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    // Find min (best)\n"
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
"    // Find max (worst)\n"
"    local_fitness[local_id] = id < pop_size ? -fitness[id] : INFINITY;\n"
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
"        worst_idx[get_group_id(0)] = local_indices[0];\n"
"    }\n"
"}\n";

// Phase function implementations
void GA_initialize_population(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                             cl_mem bounds_buffer, int dim, int pop_size, uint seed, cl_event *event) {
    cl_int err;
    DEBUG_LOG("Setting initialize kernel args: population=%p, bounds=%p, dim=%d, pop_size=%d, seed=%u\n",
              population_buffer, bounds_buffer, dim, pop_size, seed);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &pop_size);
    err |= clSetKernelArg(kernel, 4, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting initialize kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
    DEBUG_LOG("Enqueuing initialize kernel: global=%zu, local=%zu\n", global_work_size, local_work_size);
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing initialize kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

void GA_crossover(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                 cl_mem best_buffer, cl_mem bounds_buffer, int dim, int pop_size, uint seed, cl_event *event) {
    cl_int err;
    DEBUG_LOG("Setting crossover kernel args: population=%p, best=%p, bounds=%p, dim=%d, pop_size=%d, seed=%u\n",
              population_buffer, best_buffer, bounds_buffer, dim, pop_size, seed);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &best_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &dim);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &pop_size);
    err |= clSetKernelArg(kernel, 5, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting crossover kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
    DEBUG_LOG("Enqueuing crossover kernel: global=%zu, local=%zu\n", global_work_size, local_work_size);
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing crossover kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

void GA_mutate_worst(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                    cl_mem bounds_buffer, cl_mem worst_idx_buffer, int dim, int pop_size, 
                    uint seed, cl_event *event) {
    cl_int err;
    DEBUG_LOG("Setting mutate kernel args: population=%p, bounds=%p, worst_idx=%p, dim=%d, pop_size=%d, seed=%u\n",
              population_buffer, bounds_buffer, worst_idx_buffer, dim, pop_size, seed);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &worst_idx_buffer);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &dim);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &pop_size);
    err |= clSetKernelArg(kernel, 5, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting mutate kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = 64;
    size_t global_work_size = local_work_size;
    DEBUG_LOG("Enqueuing mutate kernel: global=%zu, local=%zu\n", global_work_size, local_work_size);
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing mutate kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

void GA_find_best_worst(cl_kernel kernel, cl_command_queue queue, cl_mem fitness_buffer, 
                       cl_mem best_idx_buffer, cl_mem worst_idx_buffer, int pop_size, 
                       cl_event *event) {
    cl_int err;
    DEBUG_LOG("Setting best/worst kernel args: fitness=%p, best_idx=%p, worst_idx=%p, pop_size=%d\n",
              fitness_buffer, best_idx_buffer, worst_idx_buffer, pop_size);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &fitness_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &best_idx_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &worst_idx_buffer);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &pop_size);
    err |= clSetKernelArg(kernel, 4, 64 * sizeof(float), NULL); // Local memory for fitness
    err |= clSetKernelArg(kernel, 5, 64 * sizeof(int), NULL);   // Local memory for indices
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting best/worst kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
    DEBUG_LOG("Enqueuing best/worst kernel: global=%zu, local=%zu\n", global_work_size, local_work_size);
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing best/worst kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

// Main Optimization Function
void GA_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, crossover_kernel = NULL, 
              mutate_kernel = NULL, best_worst_kernel = NULL;
    cl_mem bounds_buffer = NULL, best_buffer = NULL, best_idx_buffer = NULL, worst_idx_buffer = NULL;
    float *bounds_float = NULL, *population = NULL, *best = NULL, *fitness = NULL;
    double *position_double = NULL;
    int *best_idx_array = NULL, *worst_idx_array = NULL;
    cl_event events[4] = {0};
    double start_time, end_time;

    // Validate input
    if (!opt || !opt->population || !opt->bounds || !opt->best_solution.position || 
        !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure, OpenCL setup, or objective function\n");
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
    if (dim < 1 || pop_size < 1 || max_iter < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), pop_size (%d), or max_iter (%d)\n", 
                dim, pop_size, max_iter);
        exit(EXIT_FAILURE);
    }
    DEBUG_LOG("Optimizer parameters: dim=%d, pop_size=%d, max_iter=%d\n", dim, pop_size, max_iter);

    // Query device properties
    size_t max_work_group_size;
    err = clGetDeviceInfo(opt->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying max work group size: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = max_work_group_size < 64 ? max_work_group_size : 64;
    cl_ulong max_mem_alloc;
    err = clGetDeviceInfo(opt->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_mem_alloc, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying max mem alloc size: %d\n", err);
        exit(EXIT_FAILURE);
    }
    cl_ulong local_mem_size;
    err = clGetDeviceInfo(opt->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying local mem size: %d\n", err);
        exit(EXIT_FAILURE);
    }
    DEBUG_LOG("Device properties: max_work_group_size=%zu, max_mem_alloc=%llu, local_mem_size=%llu\n", 
              max_work_group_size, (unsigned long long)max_mem_alloc, (unsigned long long)local_mem_size);

    // Check buffer sizes
    size_t population_buffer_size = (size_t)pop_size * dim * sizeof(float);
    size_t fitness_buffer_size = (size_t)pop_size * sizeof(float);
    size_t bounds_buffer_size = (size_t)2 * dim * sizeof(float);
    if (population_buffer_size > max_mem_alloc || fitness_buffer_size > max_mem_alloc || bounds_buffer_size > max_mem_alloc) {
        fprintf(stderr, "Error: Buffer size exceeds device max memory allocation\n");
        goto cleanup;
    }
    DEBUG_LOG("Buffer sizes: population=%zu, fitness=%zu, bounds=%zu\n", 
              population_buffer_size, fitness_buffer_size, bounds_buffer_size);

    // Validate OpenCL resources
    if (!opt->context || !opt->queue || !opt->device) {
        fprintf(stderr, "Error: OpenCL context, queue, or device is null\n");
        goto cleanup;
    }
    DEBUG_LOG("OpenCL resources validated: context=%p, queue=%p, device=%p\n", 
              opt->context, opt->queue, opt->device);

    // Create program
    DEBUG_LOG("Creating OpenCL program\n");
    program = clCreateProgramWithSource(opt->context, 1, &ga_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating GA program: %d\n", err);
        goto cleanup;
    }
    DEBUG_LOG("Building OpenCL program\n");
    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size + 1);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            log[log_size] = '\0';
            fprintf(stderr, "Error building GA program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        } else {
            fprintf(stderr, "Error building GA program: %d (failed to allocate build log)\n", err);
        }
        goto cleanup;
    }
    DEBUG_LOG("Built OpenCL program\n");

    // Create kernels
    DEBUG_LOG("Creating init kernel\n");
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating init kernel: %d\n", err);
        goto cleanup;
    }
    DEBUG_LOG("Creating crossover kernel\n");
    crossover_kernel = clCreateKernel(program, "crossover", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating crossover kernel: %d\n", err);
        goto cleanup;
    }
    DEBUG_LOG("Creating mutate kernel\n");
    mutate_kernel = clCreateKernel(program, "mutate_worst", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating mutate kernel: %d\n", err);
        goto cleanup;
    }
    DEBUG_LOG("Creating best/worst kernel\n");
    best_worst_kernel = clCreateKernel(program, "find_best_worst", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating best/worst kernel: %d\n", err);
        goto cleanup;
    }
    DEBUG_LOG("Created OpenCL kernels\n");

    // Allocate host memory
    DEBUG_LOG("Allocating host memory\n");
    bounds_float = (float*)malloc(bounds_buffer_size);
    population = (float*)malloc(population_buffer_size);
    best = (float*)malloc(dim * sizeof(float));
    fitness = (float*)malloc(fitness_buffer_size);
    position_double = (double*)malloc(dim * sizeof(double));
    best_idx_array = (int*)malloc(pop_size * sizeof(int));
    worst_idx_array = (int*)malloc(pop_size * sizeof(int));
    if (!bounds_float || !population || !best || !fitness || !position_double || 
        !best_idx_array || !worst_idx_array) {
        fprintf(stderr, "Error: Host memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }
    DEBUG_LOG("Bounds: [");
    for (int i = 0; i < 2 * dim; i++) {
        DEBUG_LOG("%f", bounds_float[i]);
        if (i < 2 * dim - 1) DEBUG_LOG(", ");
    }
    DEBUG_LOG("]\n");
    opt->best_solution.fitness = INFINITY;
    DEBUG_LOG("Allocated host memory\n");

    // Create additional buffers
    DEBUG_LOG("Creating bounds buffer\n");
    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, bounds_buffer_size, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating bounds buffer: %d\n", err);
        goto cleanup;
    }
    DEBUG_LOG("Creating best buffer\n");
    best_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating best buffer: %d\n", err);
        goto cleanup;
    }
    DEBUG_LOG("Creating best idx buffer\n");
    best_idx_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating best idx buffer: %d\n", err);
        goto cleanup;
    }
    DEBUG_LOG("Creating worst idx buffer\n");
    worst_idx_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating worst idx buffer: %d\n", err);
        goto cleanup;
    }
    DEBUG_LOG("Created additional OpenCL buffers\n");

    // Write bounds
    DEBUG_LOG("Writing bounds buffer: size=%zu\n", bounds_buffer_size);
    err = clEnqueueWriteBuffer(opt->queue, bounds_buffer, CL_TRUE, 0, bounds_buffer_size, bounds_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bounds buffer: %d\n", err);
        goto cleanup;
    }

    // Process each population
    for (int p = 0; p < GA_NUM_POPULATIONS; p++) {
        uint seed = (uint)time(NULL) + p;
        int best_idx = 0, worst_idx = 0;
        DEBUG_LOG("Starting population %d\n", p + 1);

        // Initialize population
        GA_initialize_population(init_kernel, opt->queue, opt->population_buffer, bounds_buffer, 
                                dim, pop_size, seed, &events[0]);

        // Evaluate initial fitness
        DEBUG_LOG("Reading initial population for fitness: size=%zu\n", population_buffer_size);
        err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                                 population_buffer_size, population, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading population for fitness: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < pop_size; i++) {
            for (int d = 0; d < dim; d++) {
                position_double[d] = (double)population[i * dim + d];
            }
            fitness[i] = (float)objective_function(position_double);
        }
        DEBUG_LOG("Initial population and fitness:\n");
        for (int i = 0; i < pop_size; i++) {
            DEBUG_LOG("Individual %d: [", i);
            for (int d = 0; d < dim; d++) {
                DEBUG_LOG("%f", population[i * dim + d]);
                if (d < dim - 1) DEBUG_LOG(", ");
            }
            DEBUG_LOG("], Fitness: %f\n", fitness[i]);
        }
        DEBUG_LOG("Writing initial fitness: size=%zu\n", fitness_buffer_size);
        err = clEnqueueWriteBuffer(opt->queue, opt->fitness_buffer, CL_TRUE, 0, 
                                  fitness_buffer_size, fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing initial fitness: %d\n", err);
            goto cleanup;
        }

        // Find initial best
        float best_fitness = INFINITY;
        for (int i = 0; i < pop_size; i++) {
            if (fitness[i] < best_fitness) {
                best_fitness = fitness[i];
                best_idx = i;
            }
        }
        for (int d = 0; d < dim; d++) {
            best[d] = population[best_idx * dim + d];
        }
        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)best_fitness;
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)best[d];
            }
        }
        DEBUG_LOG("Writing best buffer: size=%zu\n", dim * sizeof(float));
        err = clEnqueueWriteBuffer(opt->queue, best_buffer, CL_TRUE, 0, 
                                  dim * sizeof(float), best, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing best buffer: %d\n", err);
            goto cleanup;
        }

        // Main optimization loop
        for (int iter = 0; iter < max_iter; iter++) {
            start_time = get_time_ms();

            // Evaluate fitness
            DEBUG_LOG("Reading population for fitness: size=%zu\n", population_buffer_size);
            err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                                     population_buffer_size, population, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading population for fitness: %d\n", err);
                goto cleanup;
            }
            for (int i = 0; i < pop_size; i++) {
                for (int d = 0; d < dim; d++) {
                    position_double[d] = (double)population[i * dim + d];
                }
                fitness[i] = (float)objective_function(position_double);
            }
            DEBUG_LOG("Writing fitness: size=%zu\n", fitness_buffer_size);
            err = clEnqueueWriteBuffer(opt->queue, opt->fitness_buffer, CL_TRUE, 0, 
                                      fitness_buffer_size, fitness, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing fitness: %d\n", err);
                goto cleanup;
            }

            // Find best and worst
            GA_find_best_worst(best_worst_kernel, opt->queue, opt->fitness_buffer, 
                              best_idx_buffer, worst_idx_buffer, pop_size, &events[1]);

            // Read best and worst indices
            DEBUG_LOG("Reading best/worst indices\n");
            err = clEnqueueReadBuffer(opt->queue, best_idx_buffer, CL_TRUE, 0, 
                                     sizeof(int), &best_idx, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(opt->queue, worst_idx_buffer, CL_TRUE, 0, 
                                      sizeof(int), &worst_idx, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best/worst idx buffers: %d\n", err);
                goto cleanup;
            }
            DEBUG_LOG("Best_idx=%d, Worst_idx=%d\n", best_idx, worst_idx);
            if (best_idx < 0 || best_idx >= pop_size || worst_idx < 0 || worst_idx >= pop_size) {
                fprintf(stderr, "Error: Invalid best_idx (%d) or worst_idx (%d)\n", best_idx, worst_idx);
                goto cleanup;
            }

            // Read best individual and fitness
            DEBUG_LOG("Reading best individual: idx=%d\n", best_idx);
            err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 
                                     best_idx * dim * sizeof(float), dim * sizeof(float), 
                                     best, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best individual: %d\n", err);
                goto cleanup;
            }
            err = clEnqueueReadBuffer(opt->queue, opt->fitness_buffer, CL_TRUE, 
                                     best_idx * sizeof(float), sizeof(float), 
                                     &best_fitness, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best fitness: %d\n", err);
                goto cleanup;
            }

            // Update best solution
            if (best_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = (double)best_fitness;
                for (int d = 0; d < dim; d++) {
                    opt->best_solution.position[d] = (double)best[d];
                }
                DEBUG_LOG("Updated best solution: Fitness=%f, Position=[", best_fitness);
                for (int d = 0; d < dim; d++) {
                    DEBUG_LOG("%f", best[d]);
                    if (d < dim - 1) DEBUG_LOG(", ");
                }
                DEBUG_LOG("]\n");
                err = clEnqueueWriteBuffer(opt->queue, best_buffer, CL_TRUE, 0, 
                                          dim * sizeof(float), best, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error writing best buffer: %d\n", err);
                    goto cleanup;
                }
            }

            // Perform crossover
            GA_crossover(crossover_kernel, opt->queue, opt->population_buffer, 
                        best_buffer, bounds_buffer, dim, pop_size, seed + iter, &events[2]);

            // Mutate worst
            GA_mutate_worst(mutate_kernel, opt->queue, opt->population_buffer, 
                           bounds_buffer, worst_idx_buffer, dim, pop_size, 
                           seed + iter, &events[3]);

            // Profiling
            cl_ulong time_start, time_end;
            double init_time = 0, best_worst_time = 0, crossover_time = 0, mutate_time = 0;
            for (int i = 0; i < 4; i++) {
                if (events[i]) {
                    err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, 
                                                 sizeof(cl_ulong), &time_start, NULL);
                    if (err == CL_SUCCESS) {
                        err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, 
                                                     sizeof(cl_ulong), &time_end, NULL);
                        if (err == CL_SUCCESS) {
                            double time_ms = (time_end - time_start) / 1e6;
                            if (i == 0) init_time = time_ms;
                            else if (i == 1) best_worst_time = time_ms;
                            else if (i == 2) crossover_time = time_ms;
                            else mutate_time = time_ms;
                        }
                    }
                }
            }
            end_time = get_time_ms();
            if (GA_VERBOSITY) {
                printf("GA|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Best/Worst: %.3f ms | Crossover: %.3f ms | Mutate: %.3f ms\n",
                       iter + 1, opt->best_solution.fitness, end_time - start_time, 
                       init_time, best_worst_time, crossover_time, mutate_time);
            }
        }

        if (GA_VERBOSITY) {
            printf("GA Best in Pop %d: Fitness = %e\n", p + 1, opt->best_solution.fitness);
        }
    }

    // Update CPU-side population
    DEBUG_LOG("Reading final population: size=%zu\n", population_buffer_size);
    err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                             population_buffer_size, population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final population: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < pop_size; i++) {
        for (int d = 0; d < dim; d++) {
            opt->population[i].position[d] = (double)population[i * dim + d];
        }
        opt->population[i].fitness = INFINITY; // Fitness computed on GPU, not updated here
    }
    DEBUG_LOG("Final best solution: Fitness=%f, Position=[", opt->best_solution.fitness);
    for (int d = 0; d < dim; d++) {
        DEBUG_LOG("%f", opt->best_solution.position[d]);
        if (d < dim - 1) DEBUG_LOG(", ");
    }
    DEBUG_LOG("]\n");
    DEBUG_LOG("Updated CPU-side population\n");

cleanup:
    DEBUG_LOG("Starting cleanup\n");
    if (bounds_float) free(bounds_float);
    if (population) free(population);
    if (best) free(best);
    if (fitness) free(fitness);
    if (position_double) free(position_double);
    if (best_idx_array) free(best_idx_array);
    if (worst_idx_array) free(worst_idx_array);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (best_buffer) clReleaseMemObject(best_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (worst_idx_buffer) clReleaseMemObject(worst_idx_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (crossover_kernel) clReleaseKernel(crossover_kernel);
    if (mutate_kernel) clReleaseKernel(mutate_kernel);
    if (best_worst_kernel) clReleaseKernel(best_worst_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 4; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    clFinish(opt->queue);
    DEBUG_LOG("Cleanup completed\n");
}
