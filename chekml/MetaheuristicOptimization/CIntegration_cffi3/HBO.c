#include "HBO.h"
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
static const char* hbo_kernel_source =
"#define INFINITY 1.0e30f\n"
// Simple LCG random number generator
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
"    const int population_size,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < population_size) {\n"
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
// Update population
"__kernel void update_population(\n"
"    __global float* population,\n"
"    __global float* trial_population,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        int a, b, c;\n"
"        do {\n"
"            a = lcg_rand(&local_seed) % population_size;\n"
"            b = lcg_rand(&local_seed) % population_size;\n"
"            c = lcg_rand(&local_seed) % population_size;\n"
"        } while (a == b || b == c || a == c || a == id || b == id || c == id);\n"
"        float r1 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        int idx = lcg_rand(&local_seed) % dim;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float pa = population[a * dim + d];\n"
"            float pb = population[b * dim + d];\n"
"            float pc = population[c * dim + d];\n"
"            float heap_candidate = pa + r1 * (pb - pc);\n"
"            trial_population[id * dim + d] = (d == idx) ? heap_candidate : population[id * dim + d];\n"
"            float min = local_bounds[2 * d];\n"
"            float max = local_bounds[2 * d + 1];\n"
"            trial_population[id * dim + d] = clamp(trial_population[id * dim + d], min, max);\n"
"        }\n"
"    }\n"
"}\n"
// Update population based on fitness
"__kernel void apply_trial(\n"
"    __global float* population,\n"
"    __global const float* trial_population,\n"
"    __global float* fitness,\n"
"    __global const float* trial_fitness,\n"
"    const int dim,\n"
"    const int population_size)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        if (trial_fitness[id] < fitness[id]) {\n"
"            fitness[id] = trial_fitness[id];\n"
"            for (int d = 0; d < dim; d++) {\n"
"                population[id * dim + d] = trial_population[id * dim + d];\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
// Find best solution
"__kernel void find_best(\n"
"    __global const float* fitness,\n"
"    __global int* best_idx,\n"
"    __global const float* population,\n"
"    __global float* best_position,\n"
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
"        best_idx[get_group_id(0)] = local_indices[0];\n"
"        if (get_group_id(0) == 0) {\n"
"            int idx = local_indices[0];\n"
"            for (int d = 0; d < dim; d++) {\n"
"                best_position[d] = idx >= 0 ? population[idx * dim + d] : 0.0f;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n";

void HBO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, update_kernel = NULL, apply_trial_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, population_buffer = NULL, trial_population_buffer = NULL;
    cl_mem fitness_buffer = NULL, trial_fitness_buffer = NULL, best_idx_buffer = NULL, best_position_buffer = NULL;
    float *bounds_float = NULL, *population = NULL, *trial_population = NULL;
    float *fitness = NULL, *trial_fitness = NULL, *best_position = NULL;
    int *best_idx_array = NULL;
    double *temp_position = NULL;
    cl_event events[3] = {0};
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
    const int population_size = HBO_POPULATION_SIZE;
    const int max_iter = opt->max_iter;
    if (dim < 1 || population_size < 1 || max_iter < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), or max_iter (%d)\n", 
                dim, population_size, max_iter);
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
    program = clCreateProgramWithSource(context, 1, &hbo_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating HBO program: %d\n", err);
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
            fprintf(stderr, "Error building HBO program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    update_kernel = clCreateKernel(program, "update_population", &err);
    apply_trial_kernel = clCreateKernel(program, "apply_trial", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !update_kernel || !apply_trial_kernel || !best_kernel) {
        fprintf(stderr, "Error creating HBO kernels: %d\n", err);
        goto cleanup;
    }

    // Query work-group size
    size_t max_work_group_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    size_t local_work_size = max_work_group_size < HBO_LOCAL_WORK_SIZE ? max_work_group_size : HBO_LOCAL_WORK_SIZE;

    // Create buffers
    bounds_float = (float *)malloc(2 * dim * sizeof(float));
    population = (float *)malloc(population_size * dim * sizeof(float));
    trial_population = (float *)malloc(population_size * dim * sizeof(float));
    fitness = (float *)malloc(population_size * sizeof(float));
    trial_fitness = (float *)malloc(population_size * sizeof(float));
    best_position = (float *)malloc(dim * sizeof(float));
    best_idx_array = (int *)malloc(population_size * sizeof(int));
    temp_position = (double *)malloc(dim * sizeof(double));
    if (!bounds_float || !population || !trial_population || !fitness || !trial_fitness || 
        !best_position || !best_idx_array || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }
    for (int i = 0; i < dim; i++) {
        best_position[i] = 0.0f;
    }

    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    trial_population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    trial_fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    best_position_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !population_buffer || !trial_population_buffer || 
        !fitness_buffer || !trial_fitness_buffer || !best_idx_buffer || !best_position_buffer) {
        fprintf(stderr, "Error creating HBO buffers: %d\n", err);
        goto cleanup;
    }

    // Write bounds
    err = clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bounds buffer: %d\n", err);
        goto cleanup;
    }

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = 0.0;
    }

    // Initialize population
    uint seed = (uint)time(NULL);
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 3, sizeof(int), &population_size);
    err |= clSetKernelArg(init_kernel, 4, sizeof(uint), &seed);
    err |= clSetKernelArg(init_kernel, 5, 2 * dim * sizeof(float), NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting init kernel args: %d\n", err);
        goto cleanup;
    }

    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[0]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel: %d\n", err);
        goto cleanup;
    }

    // Evaluate initial population on host
    err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, 0, 
                              population_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading population buffer: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            temp_position[d] = (double)population[i * dim + d];
        }
        fitness[i] = (float)objective_function(temp_position);
    }
    err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                               population_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing fitness buffer: %d\n", err);
        goto cleanup;
    }

    // Find initial best
    err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &fitness_buffer);
    err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &best_idx_buffer);
    err |= clSetKernelArg(best_kernel, 2, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(best_kernel, 3, sizeof(cl_mem), &best_position_buffer);
    err |= clSetKernelArg(best_kernel, 4, sizeof(int), &dim);
    err |= clSetKernelArg(best_kernel, 5, sizeof(int), &population_size);
    err |= clSetKernelArg(best_kernel, 6, local_work_size * sizeof(float), NULL);
    err |= clSetKernelArg(best_kernel, 7, local_work_size * sizeof(int), NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting best kernel args: %d\n", err);
        goto cleanup;
    }
    err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing best kernel: %d\n", err);
        goto cleanup;
    }

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Update population
        err = clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &trial_population_buffer);
        err |= clSetKernelArg(update_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(update_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(update_kernel, 4, sizeof(int), &population_size);
        err |= clSetKernelArg(update_kernel, 5, sizeof(uint), &seed);
        err |= clSetKernelArg(update_kernel, 6, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting update kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, update_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[0]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing update kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate trial population on host
        err = clEnqueueReadBuffer(queue, trial_population_buffer, CL_TRUE, 0, 
                                  population_size * dim * sizeof(float), trial_population, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading trial population buffer: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < population_size; i++) {
            for (int d = 0; d < dim; d++) {
                temp_position[d] = (double)trial_population[i * dim + d];
            }
            trial_fitness[i] = (float)objective_function(temp_position);
        }
        err = clEnqueueWriteBuffer(queue, trial_fitness_buffer, CL_TRUE, 0, 
                                   population_size * sizeof(float), trial_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing trial fitness buffer: %d\n", err);
            goto cleanup;
        }

        // Apply trial solutions
        err = clSetKernelArg(apply_trial_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(apply_trial_kernel, 1, sizeof(cl_mem), &trial_population_buffer);
        err |= clSetKernelArg(apply_trial_kernel, 2, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(apply_trial_kernel, 3, sizeof(cl_mem), &trial_fitness_buffer);
        err |= clSetKernelArg(apply_trial_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(apply_trial_kernel, 5, sizeof(int), &population_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting apply_trial kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, apply_trial_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing apply_trial kernel: %d\n", err);
            goto cleanup;
        }

        // Find best
        err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
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
        float best_fitness;
        err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, best_idx * sizeof(float), sizeof(float), &best_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best fitness: %d\n", err);
            goto cleanup;
        }
        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)best_fitness;
            err = clEnqueueReadBuffer(queue, best_position_buffer, CL_TRUE, 0, dim * sizeof(float), best_position, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best position: %d\n", err);
                goto cleanup;
            }
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)best_position[d];
            }
        }

        // Profiling
        cl_ulong time_start, time_end;
        double update_time = 0, apply_trial_time = 0, best_time = 0;
        if (events[0]) {
            clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            update_time = (time_end - time_start) / 1e6;
        }
        if (events[1]) {
            clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            apply_trial_time = (time_end - time_start) / 1e6;
        }
        if (events[2]) {
            clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            best_time = (time_end - time_start) / 1e6;
        }
        end_time = get_time_ms();
        printf("HBO|%5d -----> %9.16f | Total: %.3f ms | Update: %.3f ms | ApplyTrial: %.3f ms | Best: %.3f ms\n", 
               iter + 1, opt->best_solution.fitness, end_time - start_time, update_time, apply_trial_time, best_time);
    }

    // Update CPU-side population
    err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final population buffer: %d\n", err);
        goto cleanup;
    }
    err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final fitness buffer: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < population_size && i < opt->population_size; i++) {
        for (int d = 0; d < dim; d++) {
            opt->population[i].position[d] = (double)population[i * dim + d];
        }
        opt->population[i].fitness = (double)fitness[i];
    }

cleanup:
    if (bounds_float) free(bounds_float);
    if (population) free(population);
    if (trial_population) free(trial_population);
    if (fitness) free(fitness);
    if (trial_fitness) free(trial_fitness);
    if (best_position) free(best_position);
    if (best_idx_array) free(best_idx_array);
    if (temp_position) free(temp_position);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (population_buffer) clReleaseMemObject(population_buffer);
    if (trial_population_buffer) clReleaseMemObject(trial_population_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (trial_fitness_buffer) clReleaseMemObject(trial_fitness_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (best_position_buffer) clReleaseMemObject(best_position_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (update_kernel) clReleaseKernel(update_kernel);
    if (apply_trial_kernel) clReleaseKernel(apply_trial_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 3; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (queue) clFinish(queue);
}
