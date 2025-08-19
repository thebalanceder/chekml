#include "GWCA.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

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
static const char* gwca_kernel_source =
"#define G 9.8f\n"
"#define M 3.0f\n"
"#define E 0.1f\n"
"#define P 9.0f\n"
"#define Q 6.0f\n"
"#define CMAX 20.0f\n"
"#define CMIN 10.0f\n"
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
"    __global const float* bounds,\n"
"    __global const float* worker1_pos,\n"
"    __global const float* worker2_pos,\n"
"    __global const float* worker3_pos,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int lnp,\n"
"    const float c,\n"
"    const int iter,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id * dim + iter;\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        float r1 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float r2 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        if (id < lnp) {\n"
"            float F = (M * G * r1) / (P * Q * (1.0f + (float)iter));\n"
"            for (int d = 0; d < dim; d++) {\n"
"                int sign = lcg_rand(&local_seed) % 2 ? 1 : -1;\n"
"                population[id * dim + d] += F * (float)sign * c;\n"
"            }\n"
"        } else {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float avg_worker = (worker1_pos[d] + worker2_pos[d] + worker3_pos[d]) / 3.0f;\n"
"                population[id * dim + d] += r2 * avg_worker * c;\n"
"            }\n"
"        }\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float min = local_bounds[2 * d];\n"
"            float max = local_bounds[2 * d + 1];\n"
"            population[id * dim + d] = clamp(population[id * dim + d], min, max);\n"
"        }\n"
"    }\n"
"}\n"
// Find top three solutions
"__kernel void find_top_three(\n"
"    __global const float* fitness,\n"
"    __global int* indices,\n"
"    __global const float* population,\n"
"    __global float* worker1_pos,\n"
"    __global float* worker2_pos,\n"
"    __global float* worker3_pos,\n"
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
"        if (local_id < offset && local_fitness[local_id + offset] < local_fitness[local_id]) {\n"
"            local_fitness[local_id] = local_fitness[local_id + offset];\n"
"            local_indices[local_id] = local_indices[local_id + offset];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (local_id == 0) {\n"
"        indices[get_group_id(0)] = local_indices[0];\n"
"    }\n"
"}\n";

void GWCA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, update_kernel = NULL, top_three_kernel = NULL;
    cl_mem bounds_buffer = NULL, population_buffer = NULL, fitness_buffer = NULL;
    cl_mem worker1_buffer = NULL, worker2_buffer = NULL, worker3_buffer = NULL, indices_buffer = NULL;
    float* bounds_float = NULL, *population = NULL, *fitness = NULL;
    float* worker1_pos = NULL, *worker2_pos = NULL, *worker3_pos = NULL;
    int* indices = NULL;
    double* temp_position = NULL;
    cl_event events[2] = {0};
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
    const int population_size = GWCA_POPULATION_SIZE;
    const int max_iter = opt->max_iter;
    const int lnp = (int)ceil(population_size * E);
    if (dim < 1 || population_size < 1 || max_iter < 1 || lnp < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), max_iter (%d), or lnp (%d)\n", 
                dim, population_size, max_iter, lnp);
        exit(EXIT_FAILURE);
    }

    // Select GPU platform and device
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "Error: No OpenCL platforms found: %d\n", err);
        exit(EXIT_FAILURE);
    }
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
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
        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
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
    program = clCreateProgramWithSource(context, 1, &gwca_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating GWCA program: %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building GWCA program: %d\nBuild log:\n%s\n", err, log);
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
    top_three_kernel = clCreateKernel(program, "find_top_three", &err);
    if (err != CL_SUCCESS || !init_kernel || !update_kernel || !top_three_kernel) {
        fprintf(stderr, "Error creating GWCA kernels: %d\n", err);
        goto cleanup;
    }

    // Query work-group size
    size_t max_work_group_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    size_t local_work_size = max_work_group_size < GWCA_LOCAL_WORK_SIZE ? max_work_group_size : GWCA_LOCAL_WORK_SIZE;

    // Create buffers
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    population = (float*)malloc(population_size * dim * sizeof(float));
    fitness = (float*)malloc(population_size * sizeof(float));
    worker1_pos = (float*)malloc(dim * sizeof(float));
    worker2_pos = (float*)malloc(dim * sizeof(float));
    worker3_pos = (float*)malloc(dim * sizeof(float));
    indices = (int*)malloc(population_size * sizeof(int));
    temp_position = (double*)malloc(dim * sizeof(double));
    if (!bounds_float || !population || !fitness || !worker1_pos || !worker2_pos || !worker3_pos || !indices || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }
    for (int i = 0; i < dim; i++) {
        worker1_pos[i] = worker2_pos[i] = worker3_pos[i] = 0.0f;
    }

    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    worker1_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, dim * sizeof(float), NULL, &err);
    worker2_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, dim * sizeof(float), NULL, &err);
    worker3_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, dim * sizeof(float), NULL, &err);
    indices_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !population_buffer || !fitness_buffer || 
        !worker1_buffer || !worker2_buffer || !worker3_buffer || !indices_buffer) {
        fprintf(stderr, "Error creating GWCA buffers: %d\n", err);
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

    // Find initial top three
    err = clSetKernelArg(top_three_kernel, 0, sizeof(cl_mem), &fitness_buffer);
    err |= clSetKernelArg(top_three_kernel, 1, sizeof(cl_mem), &indices_buffer);
    err |= clSetKernelArg(top_three_kernel, 2, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(top_three_kernel, 3, sizeof(cl_mem), &worker1_buffer);
    err |= clSetKernelArg(top_three_kernel, 4, sizeof(cl_mem), &worker2_buffer);
    err |= clSetKernelArg(top_three_kernel, 5, sizeof(cl_mem), &worker3_buffer);
    err |= clSetKernelArg(top_three_kernel, 6, sizeof(int), &dim);
    err |= clSetKernelArg(top_three_kernel, 7, sizeof(int), &population_size);
    err |= clSetKernelArg(top_three_kernel, 8, local_work_size * sizeof(float), NULL);
    err |= clSetKernelArg(top_three_kernel, 9, local_work_size * sizeof(int), NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting top_three kernel args: %d\n", err);
        goto cleanup;
    }
    err = clEnqueueNDRangeKernel(queue, top_three_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing top_three kernel: %d\n", err);
        goto cleanup;
    }

    // Main optimization loop
    for (int iter = 1; iter <= max_iter; iter++) {
        start_time = get_time_ms();

        // Adjust constant C
        float c = CMAX - ((CMAX - CMIN) * iter / (float)max_iter);

        // Update worker positions
        err = clEnqueueReadBuffer(queue, indices_buffer, CL_TRUE, 0, population_size * sizeof(int), indices, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading indices buffer: %d\n", err);
            goto cleanup;
        }
        int worker1_idx = indices[0];
        int worker2_idx = indices[1 % population_size];
        int worker3_idx = indices[2 % population_size];
        if (worker1_idx >= 0 && worker1_idx < population_size) {
            err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, worker1_idx * dim * sizeof(float), 
                                      dim * sizeof(float), worker1_pos, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading worker1 position: %d\n", err);
                goto cleanup;
            }
            err = clEnqueueWriteBuffer(queue, worker1_buffer, CL_TRUE, 0, dim * sizeof(float), worker1_pos, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing worker1 buffer: %d\n", err);
                goto cleanup;
            }
        }
        if (worker2_idx >= 0 && worker2_idx < population_size) {
            err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, worker2_idx * dim * sizeof(float), 
                                      dim * sizeof(float), worker2_pos, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading worker2 position: %d\n", err);
                goto cleanup;
            }
            err = clEnqueueWriteBuffer(queue, worker2_buffer, CL_TRUE, 0, dim * sizeof(float), worker2_pos, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing worker2 buffer: %d\n", err);
                goto cleanup;
            }
        }
        if (worker3_idx >= 0 && worker3_idx < population_size) {
            err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, worker3_idx * dim * sizeof(float), 
                                      dim * sizeof(float), worker3_pos, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading worker3 position: %d\n", err);
                goto cleanup;
            }
            err = clEnqueueWriteBuffer(queue, worker3_buffer, CL_TRUE, 0, dim * sizeof(float), worker3_pos, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing worker3 buffer: %d\n", err);
                goto cleanup;
            }
        }

        // Update population
        err = clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(update_kernel, 2, sizeof(cl_mem), &worker1_buffer);
        err |= clSetKernelArg(update_kernel, 3, sizeof(cl_mem), &worker2_buffer);
        err |= clSetKernelArg(update_kernel, 4, sizeof(cl_mem), &worker3_buffer);
        err |= clSetKernelArg(update_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(update_kernel, 6, sizeof(int), &population_size);
        err |= clSetKernelArg(update_kernel, 7, sizeof(int), &lnp);
        err |= clSetKernelArg(update_kernel, 8, sizeof(float), &c);
        err |= clSetKernelArg(update_kernel, 9, sizeof(int), &iter);
        err |= clSetKernelArg(update_kernel, 10, sizeof(uint), &seed);
        err |= clSetKernelArg(update_kernel, 11, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting update kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, update_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[0]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing update kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate population on host
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

        // Find top three
        err = clEnqueueNDRangeKernel(queue, top_three_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing top_three kernel: %d\n", err);
            goto cleanup;
        }

        // Update best solution
        err = clEnqueueReadBuffer(queue, indices_buffer, CL_TRUE, 0, population_size * sizeof(int), indices, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading indices buffer: %d\n", err);
            goto cleanup;
        }
        int best_idx = indices[0];
        float best_fitness;
        err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, best_idx * sizeof(float), sizeof(float), &best_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best fitness: %d\n", err);
            goto cleanup;
        }
        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)best_fitness;
            err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, best_idx * dim * sizeof(float), 
                                      dim * sizeof(float), worker1_pos, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best position: %d\n", err);
                goto cleanup;
            }
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)worker1_pos[d];
            }
        }

        // Profiling
        cl_ulong time_start, time_end;
        double update_time = 0, top_three_time = 0;
        if (events[0]) {
            clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            update_time = (time_end - time_start) / 1e6;
        }
        if (events[1]) {
            clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            top_three_time = (time_end - time_start) / 1e6;
        }
        end_time = get_time_ms();
        printf("GWCA|%5d -----> %9.16f | Total: %.3f ms | Update: %.3f ms | TopThree: %.3f ms\n", 
               iter, opt->best_solution.fitness, end_time - start_time, update_time, top_three_time);
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
    if (fitness) free(fitness);
    if (worker1_pos) free(worker1_pos);
    if (worker2_pos) free(worker2_pos);
    if (worker3_pos) free(worker3_pos);
    if (indices) free(indices);
    if (temp_position) free(temp_position);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (population_buffer) clReleaseMemObject(population_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (worker1_buffer) clReleaseMemObject(worker1_buffer);
    if (worker2_buffer) clReleaseMemObject(worker2_buffer);
    if (worker3_buffer) clReleaseMemObject(worker3_buffer);
    if (indices_buffer) clReleaseMemObject(indices_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (update_kernel) clReleaseKernel(update_kernel);
    if (top_three_kernel) clReleaseKernel(top_three_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 2; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (queue) clFinish(queue);
}
