/* MA.c - GPU-Optimized Memetic Algorithm with Flexible Objective Function */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
#include "generaloptimizer.h"
#include "MA.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Algorithm parameters
#define MA_MUTATION_RATE 0.1f
#define MA_CROSSOVER_RATE 0.8f
#define MA_LOCAL_SEARCH_RATE 0.1f
#define MA_LOCAL_SEARCH_ITERS 100
#define MA_MUTATION_STDDEV 0.1f

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
static const char* ma_kernel_source =
"// Simple LCG random number generator\n"
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
"// Normal distribution approximation (Box-Muller)\n"
"float rand_normal(uint* seed, float stddev) {\n"
"    float u1 = lcg_rand_float(seed, 0.0f, 1.0f);\n"
"    float u2 = lcg_rand_float(seed, 0.0f, 1.0f);\n"
"    return stddev * sqrt(-2.0f * log(u1)) * cos(2.0f * 3.14159265359f * u2);\n"
"}\n"
"// Clamp solution to bounds\n"
"void clamp(__global float* vec, const int dim, __local const float* bounds) {\n"
"    for (int i = 0; i < dim; i++) {\n"
"        vec[i] = max(bounds[2 * i], min(bounds[2 * i + 1], vec[i]));\n"
"    }\n"
"}\n"
"// Check if a point is within bounds\n"
"bool in_bounds(__global const float* point, __local const float* bounds, const int dim) {\n"
"    for (int i = 0; i < dim; i++) {\n"
"        if (point[i] < bounds[2 * i] || point[i] > bounds[2 * i + 1]) {\n"
"            return false;\n"
"        }\n"
"    }\n"
"    return true;\n"
"}\n"
"// Placeholder for objective function (user must implement)\n"
"float evaluate_fitness(__global const float* solution, const int dim) {\n"
"    // Example: Sphere function (sum of squares)\n"
"    float sum = 0.0f;\n"
"    for (int i = 0; i < dim; i++) {\n"
"        sum += solution[i] * solution[i];\n"
"    }\n"
"    return sum;\n"
"}\n"
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
"    }\n"
"}\n"
"__kernel void evaluate_population(\n"
"    __global const float* population,\n"
"    __global float* fitness,\n"
"    const int dim,\n"
"    const int pop_size)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < pop_size) {\n"
"        fitness[id] = evaluate_fitness(&population[id * dim], dim);\n"
"    }\n"
"}\n"
"#define MA_LOCAL_SEARCH_ITERS 100\n"
"__kernel void hill_climb(\n"
"    __global float* population,\n"
"    __global float* fitness,\n"
"    __global float* candidates,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    const int ls_count,\n"
"    const float stddev,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < ls_count) {\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        uint local_seed = seed + id * dim;\n"
"        int pop_idx = lcg_rand(&local_seed) % pop_size;\n"
"        float best_fit = fitness[pop_idx];\n"
"        for (int d = 0; d < dim; d++) {\n"
"            candidates[id * dim + d] = population[pop_idx * dim + d];\n"
"        }\n"
"        for (int it = 0; it < MA_LOCAL_SEARCH_ITERS; it++) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                candidates[id * dim + d] = population[pop_idx * dim + d] + rand_normal(&local_seed, stddev);\n"
"            }\n"
"            clamp(&candidates[id * dim], dim, local_bounds);\n"
"            float fit = evaluate_fitness(&candidates[id * dim], dim);\n"
"            if (fit < best_fit) {\n"
"                for (int d = 0; d < dim; d++) {\n"
"                    population[pop_idx * dim + d] = candidates[id * dim + d];\n"
"                }\n"
"                best_fit = fit;\n"
"            }\n"
"        }\n"
"        fitness[pop_idx] = best_fit;\n"
"    }\n"
"}\n"
"__kernel void crossover(\n"
"    __global const float* population,\n"
"    __global float* children,\n"
"    __global const int* indices,\n"
"    const int dim,\n"
"    const int num_children,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < num_children / 2) {\n"
"        uint local_seed = seed + id;\n"
"        int p1 = indices[2 * id];\n"
"        int p2 = indices[2 * id + 1];\n"
"        int cx = lcg_rand(&local_seed) % dim;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            children[(2 * id) * dim + d] = (d <= cx) ? population[p1 * dim + d] : population[p2 * dim + d];\n"
"            children[(2 * id + 1) * dim + d] = (d <= cx) ? population[p2 * dim + d] : population[p1 * dim + d];\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void mutate(\n"
"    __global float* children,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int num_children,\n"
"    const float stddev,\n"
"    const float mutation_rate,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < num_children) {\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        uint local_seed = seed + id;\n"
"        if (lcg_rand_float(&local_seed, 0.0f, 1.0f) < mutation_rate) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                children[id * dim + d] += rand_normal(&local_seed, stddev);\n"
"            }\n"
"            clamp(&children[id * dim], dim, local_bounds);\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void replace_worst(\n"
"    __global float* population,\n"
"    __global float* fitness,\n"
"    __global const float* children,\n"
"    __global const float* child_fitness,\n"
"    __global const int* indices,\n"
"    const int dim,\n"
"    const int num_children,\n"
"    const int pop_size)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < num_children) {\n"
"        int worst_idx = indices[pop_size - 1 - id];\n"
"        for (int d = 0; d < dim; d++) {\n"
"            population[worst_idx * dim + d] = children[id * dim + d];\n"
"        }\n"
"        fitness[worst_idx] = child_fitness[id];\n"
"    }\n"
"}\n"
"__kernel void find_best(\n"
"    __global const float* fitness,\n"
"    __global int* best_idx,\n"
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

void MA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, eval_kernel = NULL, hill_climb_kernel = NULL, crossover_kernel = NULL, mutate_kernel = NULL, replace_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, population_buffer = NULL, fitness_buffer = NULL, candidates_buffer = NULL, children_buffer = NULL, child_fitness_buffer = NULL, indices_buffer = NULL, best_idx_buffer = NULL;
    float *bounds_float = NULL, *population = NULL, *fitness = NULL, *candidates = NULL, *children = NULL, *child_fitness = NULL;
    int *indices = NULL, *best_idx_array = NULL;
    cl_event events[6] = {0};
    double start_time, end_time;
    cl_context context = NULL;
    cl_command_queue queue = NULL;

    // Validate input
    if (!opt || !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        return;
    }
    const int dim = opt->dim;
    const int pop_size = 256; // Fixed for GPU scalability
    const int max_iter = 1000;
    if (dim < 1 || dim > 32) {
        fprintf(stderr, "Error: Invalid dim (%d), must be between 1 and 32\n", dim);
        return;
    }

    // Select GPU platform and device
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "Error: No OpenCL platforms found: %d\n", err);
        return;
    }
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) {
        fprintf(stderr, "Error: Memory allocation failed for platforms\n");
        return;
    }
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error getting platform IDs: %d\n", err);
        free(platforms);
        return;
    }

    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) continue;
        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        if (!devices) continue;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
        if (err != CL_SUCCESS) {
            free(devices);
            continue;
        }
        platform = platforms[i];
        device = devices[0]; // Use first available GPU
        free(devices);
        break;
    }
    free(platforms);
    if (!platform || !device) {
        fprintf(stderr, "Error: No suitable GPU device found\n");
        return;
    }

    // Create context and queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !context) {
        fprintf(stderr, "Error creating OpenCL context: %d\n", err);
        return;
    }
    cl_queue_properties queue_props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, queue_props, &err);
    if (err != CL_SUCCESS || !queue) {
        fprintf(stderr, "Error creating OpenCL queue: %d\n", err);
        clReleaseContext(context);
        return;
    }

    // Query device capabilities
    size_t max_work_group_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying CL_DEVICE_MAX_WORK_GROUP_SIZE: %d\n", err);
        goto cleanup;
    }
    size_t local_work_size = max_work_group_size < 64 ? max_work_group_size : 64;

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = 0.0;
    }

    // Create OpenCL program
    program = clCreateProgramWithSource(context, 1, &ma_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating MA program: %d\n", err);
        goto cleanup;
    }
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building MA program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        program = NULL;
        goto cleanup;
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    eval_kernel = clCreateKernel(program, "evaluate_population", &err);
    hill_climb_kernel = clCreateKernel(program, "hill_climb", &err);
    crossover_kernel = clCreateKernel(program, "crossover", &err);
    mutate_kernel = clCreateKernel(program, "mutate", &err);
    replace_kernel = clCreateKernel(program, "replace_worst", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !eval_kernel || !hill_climb_kernel || !crossover_kernel || !mutate_kernel || !replace_kernel || !best_kernel) {
        fprintf(stderr, "Error creating MA kernels: %d\n", err);
        goto cleanup;
    }

    // Allocate host memory
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    population = (float*)malloc(pop_size * dim * sizeof(float));
    fitness = (float*)malloc(pop_size * sizeof(float));
    candidates = (float*)malloc(pop_size * dim * sizeof(float));
    children = (float*)malloc(pop_size * dim * sizeof(float));
    child_fitness = (float*)malloc(pop_size * sizeof(float));
    indices = (int*)malloc(pop_size * sizeof(int));
    best_idx_array = (int*)malloc(pop_size * sizeof(int));
    if (!bounds_float || !population || !fitness || !candidates || !children || !child_fitness || !indices || !best_idx_array) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }

    // Create buffers
    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * sizeof(float), NULL, &err);
    candidates_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    children_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    child_fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * sizeof(float), NULL, &err);
    indices_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * sizeof(int), NULL, &err);
    best_idx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !population_buffer || !fitness_buffer || !candidates_buffer || 
        !children_buffer || !child_fitness_buffer || !indices_buffer || !best_idx_buffer) {
        fprintf(stderr, "Error creating MA buffers: %d\n", err);
        goto cleanup;
    }

    // Write bounds
    err = clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bounds buffer: %d\n", err);
        goto cleanup;
    }

    uint seed = (uint)time(NULL);
    const int ls_count = (int)(MA_LOCAL_SEARCH_RATE * pop_size);
    const int num_children = ((int)(MA_CROSSOVER_RATE * pop_size)) & ~1; // Ensure even

    // Main MA loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Initialize population
        err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(init_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(init_kernel, 3, sizeof(int), &pop_size);
        err |= clSetKernelArg(init_kernel, 4, sizeof(uint), &seed);
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
        err = clSetKernelArg(eval_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(eval_kernel, 1, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(eval_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(eval_kernel, 3, sizeof(int), &pop_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting eval kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, eval_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing eval kernel: %d\n", err);
            goto cleanup;
        }

        // Hill climbing
        float stddev = MA_MUTATION_STDDEV;
        err = clSetKernelArg(hill_climb_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(hill_climb_kernel, 1, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(hill_climb_kernel, 2, sizeof(cl_mem), &candidates_buffer);
        err |= clSetKernelArg(hill_climb_kernel, 3, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(hill_climb_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(hill_climb_kernel, 5, sizeof(int), &pop_size);
        err |= clSetKernelArg(hill_climb_kernel, 6, sizeof(int), &ls_count);
        err |= clSetKernelArg(hill_climb_kernel, 7, sizeof(float), &stddev);
        err |= clSetKernelArg(hill_climb_kernel, 8, sizeof(uint), &seed);
        err |= clSetKernelArg(hill_climb_kernel, 9, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting hill climb kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((ls_count + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, hill_climb_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing hill climb kernel: %d\n", err);
            goto cleanup;
        }

        // Selection: Sort indices on CPU (temporary, could be GPU-accelerated)
        err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, 0, pop_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading fitness buffer: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < pop_size; i++) indices[i] = i;
        for (int i = 0; i < pop_size - 1; i++) {
            for (int j = i + 1; j < pop_size; j++) {
                if (fitness[indices[i]] > fitness[indices[j]]) {
                    int tmp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                }
            }
        }
        err = clEnqueueWriteBuffer(queue, indices_buffer, CL_TRUE, 0, pop_size * sizeof(int), indices, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing indices buffer: %d\n", err);
            goto cleanup;
        }

        // Crossover
        err = clSetKernelArg(crossover_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(crossover_kernel, 1, sizeof(cl_mem), &children_buffer);
        err |= clSetKernelArg(crossover_kernel, 2, sizeof(cl_mem), &indices_buffer);
        err |= clSetKernelArg(crossover_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(crossover_kernel, 4, sizeof(int), &num_children);
        err |= clSetKernelArg(crossover_kernel, 5, sizeof(uint), &seed);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting crossover kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((num_children / 2 + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, crossover_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[3]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing crossover kernel: %d\n", err);
            goto cleanup;
        }

        // Mutation
        float mutation_rate = MA_MUTATION_RATE;
        err = clSetKernelArg(mutate_kernel, 0, sizeof(cl_mem), &children_buffer);
        err |= clSetKernelArg(mutate_kernel, 1, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(mutate_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(mutate_kernel, 3, sizeof(int), &num_children);
        err |= clSetKernelArg(mutate_kernel, 4, sizeof(float), &stddev);
        err |= clSetKernelArg(mutate_kernel, 5, sizeof(float), &mutation_rate);
        err |= clSetKernelArg(mutate_kernel, 6, sizeof(uint), &seed);
        err |= clSetKernelArg(mutate_kernel, 7, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting mutate kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((num_children + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, mutate_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[4]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing mutate kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate children
        err = clSetKernelArg(eval_kernel, 0, sizeof(cl_mem), &children_buffer);
        err |= clSetKernelArg(eval_kernel, 1, sizeof(cl_mem), &child_fitness_buffer);
        err |= clSetKernelArg(eval_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(eval_kernel, 3, sizeof(int), &num_children);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting eval kernel args for children: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((num_children + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, eval_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[5]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing eval kernel for children: %d\n", err);
            goto cleanup;
        }

        // Replace worst
        err = clSetKernelArg(replace_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(replace_kernel, 1, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(replace_kernel, 2, sizeof(cl_mem), &children_buffer);
        err |= clSetKernelArg(replace_kernel, 3, sizeof(cl_mem), &child_fitness_buffer);
        err |= clSetKernelArg(replace_kernel, 4, sizeof(cl_mem), &indices_buffer);
        err |= clSetKernelArg(replace_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(replace_kernel, 6, sizeof(int), &num_children);
        err |= clSetKernelArg(replace_kernel, 7, sizeof(int), &pop_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting replace kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((num_children + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, replace_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing replace kernel: %d\n", err);
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
        err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing best kernel: %d\n", err);
            goto cleanup;
        }

        // Update best solution
        int best_idx;
        float best_fitness;
        err = clEnqueueReadBuffer(queue, best_idx_buffer, CL_TRUE, 0, sizeof(int), &best_idx, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best_idx buffer: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, best_idx * sizeof(float), sizeof(float), &best_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best fitness: %d\n", err);
            goto cleanup;
        }
        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)best_fitness;
            err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, best_idx * dim * sizeof(float), dim * sizeof(float), population, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best solution: %d\n", err);
                goto cleanup;
            }
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)population[d];
            }
        }

        // Profiling
        cl_ulong time_start, time_end;
        double init_time = 0, eval_time = 0, hill_climb_time = 0, crossover_time = 0, mutate_time = 0, eval_child_time = 0;
        cl_ulong queue_properties;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(cl_ulong), &queue_properties, NULL);
        if (err == CL_SUCCESS && (queue_properties & CL_QUEUE_PROFILING_ENABLE)) {
            for (int i = 0; i < 6; i++) {
                if (events[i]) {
                    err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                    if (err == CL_SUCCESS) {
                        err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                        if (err == CL_SUCCESS) {
                            double time_ms = (time_end - time_start) / 1e6;
                            if (i == 0) init_time = time_ms;
                            else if (i == 1) eval_time = time_ms;
                            else if (i == 2) hill_climb_time = time_ms;
                            else if (i == 3) crossover_time = time_ms;
                            else if (i == 4) mutate_time = time_ms;
                            else if (i == 5) eval_child_time = time_ms;
                        }
                    }
                }
            }
            end_time = get_time_ms();
            printf("MA | Iteration %4d -> Best Fitness = %.16f | Total: %.3f ms | Init: %.3f ms | Eval: %.3f ms | HillClimb: %.3f ms | Crossover: %.3f ms | Mutate: %.3f ms | EvalChild: %.3f ms\n",
                   iter + 1, opt->best_solution.fitness, end_time - start_time, init_time, eval_time, hill_climb_time, crossover_time, mutate_time, eval_child_time);
        } else {
            end_time = get_time_ms();
            printf("MA | Iteration %4d -> Best Fitness = %.16f | Total: %.3f ms | Profiling disabled\n",
                   iter + 1, opt->best_solution.fitness, end_time - start_time);
        }
    }

cleanup:
    if (bounds_float) free(bounds_float);
    if (population) free(population);
    if (fitness) free(fitness);
    if (candidates) free(candidates);
    if (children) free(children);
    if (child_fitness) free(child_fitness);
    if (indices) free(indices);
    if (best_idx_array) free(best_idx_array);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (population_buffer) clReleaseMemObject(population_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (candidates_buffer) clReleaseMemObject(candidates_buffer);
    if (children_buffer) clReleaseMemObject(children_buffer);
    if (child_fitness_buffer) clReleaseMemObject(child_fitness_buffer);
    if (indices_buffer) clReleaseMemObject(indices_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (eval_kernel) clReleaseKernel(eval_kernel);
    if (hill_climb_kernel) clReleaseKernel(hill_climb_kernel);
    if (crossover_kernel) clReleaseKernel(crossover_kernel);
    if (mutate_kernel) clReleaseKernel(mutate_kernel);
    if (replace_kernel) clReleaseKernel(replace_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 6; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (queue) clFinish(queue);
}
