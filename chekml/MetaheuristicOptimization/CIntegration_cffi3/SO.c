/* SO.c - GPU-Optimized Spiral Optimization (SO) Algorithm */
#include "SO.h"
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
static const char* so_kernel_source =
"#define SPIRAL_STEP 0.1f\n"
// Simple LCG random number generator
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
"    const int population_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            population[id * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"        }\n"
"        enforce_bounds(&population[id * dim], bounds, dim);\n"
"    }\n"
"}\n"
// Spiral Movement Kernel
"__kernel void spiral_movement_phase(\n"
"    __global float* population,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        float r = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float theta = 2.0f * 3.1415926535f * lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float ct = cos(theta);\n"
"        float st = sin(theta);\n"
"        if (dim == 2) {\n"
"            population[id * dim + 0] += SPIRAL_STEP * r * ct;\n"
"            population[id * dim + 1] += SPIRAL_STEP * r * st;\n"
"        } else {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float dir = d < 2 ? (d == 0 ? ct : st) : lcg_rand_float(&local_seed, -1.0f, 1.0f);\n"
"                population[id * dim + d] += SPIRAL_STEP * r * dir;\n"
"            }\n"
"        }\n"
"        enforce_bounds(&population[id * dim], bounds, dim);\n"
"    }\n"
"}\n"
// Find Best Kernel
"__kernel void find_best(\n"
"    __global const float* fitness,\n"
"    __global int* best_idx,\n"
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
"    }\n"
"}\n";

// Phase function implementations
void SO_initialize_population(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                             cl_mem bounds_buffer, int dim, int population_size, uint seed, cl_event *event) {
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &population_size);
    err |= clSetKernelArg(kernel, 4, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting initialize kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing initialize kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

void SO_spiral_movement_phase(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                             cl_mem bounds_buffer, int dim, int population_size, uint seed, cl_event *event) {
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &population_size);
    err |= clSetKernelArg(kernel, 4, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting spiral movement kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing spiral movement kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

void SO_find_best(cl_kernel kernel, cl_command_queue queue, cl_mem fitness_buffer, 
                  cl_mem best_idx_buffer, int population_size, cl_event *event) {
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &fitness_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &best_idx_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &population_size);
    err |= clSetKernelArg(kernel, 3, 64 * sizeof(float), NULL);
    err |= clSetKernelArg(kernel, 4, 64 * sizeof(int), NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting find best kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing find best kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

// Main Optimization Function
void SO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, spiral_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, best_idx_buffer = NULL;
    float *bounds_float = NULL, *population = NULL, *fitness = NULL;
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
    const int population_size = opt->population_size;
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
    program = clCreateProgramWithSource(context, 1, &so_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating SO program: %d\n", err);
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
            fprintf(stderr, "Error building SO program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    spiral_kernel = clCreateKernel(program, "spiral_movement_phase", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !spiral_kernel || !best_kernel) {
        fprintf(stderr, "Error creating SO kernels: %d\n", err);
        goto cleanup;
    }

    // Create buffers
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    population = (float*)malloc(population_size * dim * sizeof(float));
    fitness = (float*)malloc(population_size * sizeof(float));
    best_idx_array = (int*)malloc(population_size * sizeof(int));
    temp_position = (double*)malloc(dim * sizeof(double));
    if (!bounds_float || !population || !fitness || !best_idx_array || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }
    opt->best_solution.fitness = INFINITY;

    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    opt->population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    opt->fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !best_idx_buffer || !opt->population_buffer || !opt->fitness_buffer) {
        fprintf(stderr, "Error creating SO buffers: %d\n", err);
        goto cleanup;
    }

    // Write initial bounds
    err = clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bounds buffer: %d\n", err);
        goto cleanup;
    }

    // Initialize population
    uint seed = (uint)time(NULL);
    SO_initialize_population(init_kernel, queue, opt->population_buffer, bounds_buffer, dim, population_size, seed, &events[0]);

    // Evaluate initial fitness on host and set initial best
    err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading population buffer: %d\n", err);
        goto cleanup;
    }
    float best_fitness = INFINITY;
    int best_idx = 0;
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            temp_position[d] = (double)population[i * dim + d];
        }
        fitness[i] = (float)objective_function(temp_position);
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            best_idx = i;
        }
    }
    err = clEnqueueWriteBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing fitness buffer: %d\n", err);
        goto cleanup;
    }
    opt->best_solution.fitness = (double)best_fitness;
    for (int d = 0; d < dim; d++) {
        opt->best_solution.position[d] = (double)population[best_idx * dim + d];
    }

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Spiral movement
        SO_spiral_movement_phase(spiral_kernel, queue, opt->population_buffer, bounds_buffer, dim, population_size, seed, &events[1]);

        // Evaluate fitness on host
        err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
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
        err = clEnqueueWriteBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            goto cleanup;
        }

        // Find best
        SO_find_best(best_kernel, queue, opt->fitness_buffer, best_idx_buffer, population_size, &events[2]);

        // Update best solution
        err = clEnqueueReadBuffer(queue, best_idx_buffer, CL_TRUE, 0, sizeof(int), &best_idx, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best_idx buffer: %d\n", err);
            goto cleanup;
        }
        best_fitness = fitness[best_idx];
        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)best_fitness;
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)population[best_idx * dim + d];
            }
        }

        // Sort population by fitness on host
        typedef struct { float fitness; int index; } IndexedSolution;
        IndexedSolution *indexed = (IndexedSolution*)malloc(population_size * sizeof(IndexedSolution));
        if (!indexed) {
            fprintf(stderr, "Error: Memory allocation failed for indexed\n");
            goto cleanup;
        }
        for (int i = 0; i < population_size; i++) {
            indexed[i].fitness = fitness[i];
            indexed[i].index = i;
        }
        for (int i = 0; i < population_size - 1; i++) {
            for (int j = 0; j < population_size - i - 1; j++) {
                if (indexed[j].fitness > indexed[j + 1].fitness) {
                    IndexedSolution temp = indexed[j];
                    indexed[j] = indexed[j + 1];
                    indexed[j + 1] = temp;
                }
            }
        }
        float *temp_pop = (float*)malloc(population_size * dim * sizeof(float));
        if (!temp_pop) {
            fprintf(stderr, "Error: Memory allocation failed for temp_pop\n");
            free(indexed);
            goto cleanup;
        }
        for (int i = 0; i < population_size; i++) {
            int old_idx = indexed[i].index;
            for (int d = 0; d < dim; d++) {
                temp_pop[i * dim + d] = population[old_idx * dim + d];
            }
            fitness[i] = indexed[i].fitness;
        }
        memcpy(population, temp_pop, population_size * dim * sizeof(float));
        err = clEnqueueWriteBuffer(queue, opt->population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing population buffer: %d\n", err);
            free(indexed);
            free(temp_pop);
            goto cleanup;
        }
        free(indexed);
        free(temp_pop);

        // Profiling
        cl_ulong time_start, time_end;
        double init_time = 0, spiral_time = 0, best_time = 0;
        for (int i = 0; i < 3; i++) {
            if (events[i]) {
                err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                    if (err == CL_SUCCESS) {
                        double time_ms = (time_end - time_start) / 1e6;
                        if (i == 0) init_time = time_ms;
                        else if (i == 1) spiral_time = time_ms;
                        else best_time = time_ms;
                    }
                }
            }
        }
        end_time = get_time_ms();
        printf("SO|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Spiral: %.3f ms | Best: %.3f ms\n",
               iter + 1, opt->best_solution.fitness, end_time - start_time, 
               init_time, spiral_time, best_time);
    }

    // Update CPU-side population
    err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final population buffer: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            opt->population[i].position[d] = (double)population[i * dim + d];
        }
        for (int d = 0; d < dim; d++) {
            temp_position[d] = (double)population[i * dim + d];
        }
        opt->population[i].fitness = objective_function(temp_position);
    }

cleanup:
    if (bounds_float) free(bounds_float);
    if (population) free(population);
    if (fitness) free(fitness);
    if (best_idx_array) free(best_idx_array);
    if (temp_position) free(temp_position);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (opt->population_buffer) clReleaseMemObject(opt->population_buffer);
    if (opt->fitness_buffer) clReleaseMemObject(opt->fitness_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (spiral_kernel) clReleaseKernel(spiral_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 3; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (queue) clFinish(queue);
}
