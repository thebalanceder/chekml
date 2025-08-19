#include "VNS.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

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

// OpenCL kernel source for VNS
static const char* vns_kernel_source =
"#define MUTATION_RATE 0.1f\n"
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
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
"    }\n"
"}\n"
"__kernel void generate_neighbors(\n"
"    __global const float* population,\n"
"    __global float* neighbors,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int neighborhood_size,\n"
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
"            float current = population[id * dim + d];\n"
"            float lower = local_bounds[2 * d];\n"
"            float upper = local_bounds[2 * d + 1];\n"
"            float range = upper - lower;\n"
"            float mutation = MUTATION_RATE * range * lcg_rand_float(&local_seed, -0.5f, 0.5f) * neighborhood_size;\n"
"            float value = current + mutation;\n"
"            value = value < lower ? lower : (value > upper ? upper : value);\n"
"            neighbors[id * dim + d] = value;\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void update_population(\n"
"    __global float* population,\n"
"    __global float* population_fitness,\n"
"    __global const float* neighbors,\n"
"    __global const float* neighbor_fitness,\n"
"    const int dim,\n"
"    const int population_size)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        if (neighbor_fitness[id] < population_fitness[id]) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                population[id * dim + d] = neighbors[id * dim + d];\n"
"            }\n"
"            population_fitness[id] = neighbor_fitness[id];\n"
"        }\n"
"    }\n"
"}\n"
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

void VNS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, neighbor_kernel = NULL, update_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, neighbor_buffer = NULL, neighbor_fitness_buffer = NULL, best_idx_buffer = NULL;
    float* bounds_float = NULL, *population = NULL, *fitness = NULL, *neighbors = NULL, *neighbor_fitness = NULL;
    int* best_idx_array = NULL;
    cl_event events[4] = {0};
    double start_time, end_time;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    double* temp_position = NULL;

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
    if (dim > 32) {
        fprintf(stderr, "Error: dim (%d) exceeds maximum supported value of 32\n", dim);
        exit(EXIT_FAILURE);
    }

    // Neighborhood sizes
    const int neighborhood_sizes[] = {1, 2, 3, 4};
    const int num_neighborhoods = 4;

    // Select GPU platform and device
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
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
        device = devices[0]; // Select first GPU
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
    if (err != CL_SUCCESS || !context) {
        fprintf(stderr, "Error creating OpenCL context: %d\n", err);
        exit(EXIT_FAILURE);
    }
    cl_queue_properties queue_props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, queue_props, &err);
    if (err != CL_SUCCESS || !queue) {
        fprintf(stderr, "Error creating OpenCL queue: %d\n", err);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
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
    program = clCreateProgramWithSource(context, 1, &vns_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating VNS program: %d\n", err);
        goto cleanup;
    }
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building VNS program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        goto cleanup;
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    neighbor_kernel = clCreateKernel(program, "generate_neighbors", &err);
    update_kernel = clCreateKernel(program, "update_population", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !neighbor_kernel || !update_kernel || !best_kernel) {
        fprintf(stderr, "Error creating VNS kernels: %d\n", err);
        goto cleanup;
    }

    // Create buffers
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    population = (float*)malloc(population_size * dim * sizeof(float));
    fitness = (float*)malloc(population_size * sizeof(float));
    neighbors = (float*)malloc(population_size * dim * sizeof(float));
    neighbor_fitness = (float*)malloc(population_size * sizeof(float));
    best_idx_array = (int*)malloc(population_size * sizeof(int));
    temp_position = (double*)malloc(dim * sizeof(double));
    if (!bounds_float || !population || !fitness || !neighbors || !neighbor_fitness || !best_idx_array || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }

    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    neighbor_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    neighbor_fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    opt->population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    opt->fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !neighbor_buffer || !neighbor_fitness_buffer || 
        !best_idx_buffer || !opt->population_buffer || !opt->fitness_buffer) {
        fprintf(stderr, "Error creating VNS buffers: %d\n", err);
        goto cleanup;
    }

    // Write bounds
    err = clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bounds buffer: %d\n", err);
        goto cleanup;
    }

    // Initialize population
    uint seed = (uint)time(NULL);
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 3, sizeof(int), &population_size);
    err |= clSetKernelArg(init_kernel, 4, sizeof(uint), &seed);
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

    // Evaluate initial population fitness on host
    err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, 
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
    err = clEnqueueWriteBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, 
                               population_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing fitness buffer: %d\n", err);
        goto cleanup;
    }

    // Main VNS loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        for (int k = 0; k < num_neighborhoods; k++) {
            int neighborhood_size = neighborhood_sizes[k];

            // Generate neighbors
            err = clSetKernelArg(neighbor_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
            err |= clSetKernelArg(neighbor_kernel, 1, sizeof(cl_mem), &neighbor_buffer);
            err |= clSetKernelArg(neighbor_kernel, 2, sizeof(cl_mem), &bounds_buffer);
            err |= clSetKernelArg(neighbor_kernel, 3, sizeof(int), &dim);
            err |= clSetKernelArg(neighbor_kernel, 4, sizeof(int), &population_size);
            err |= clSetKernelArg(neighbor_kernel, 5, sizeof(int), &neighborhood_size);
            err |= clSetKernelArg(neighbor_kernel, 6, sizeof(uint), &seed);
            err |= clSetKernelArg(neighbor_kernel, 7, 2 * dim * sizeof(float), NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error setting neighbor kernel args: %d\n", err);
                goto cleanup;
            }

            err = clEnqueueNDRangeKernel(queue, neighbor_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error enqueuing neighbor kernel: %d\n", err);
                goto cleanup;
            }

            // Evaluate neighbor fitness on host
            err = clEnqueueReadBuffer(queue, neighbor_buffer, CL_TRUE, 0, 
                                      population_size * dim * sizeof(float), neighbors, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading neighbor buffer: %d\n", err);
                goto cleanup;
            }
            for (int i = 0; i < population_size; i++) {
                for (int d = 0; d < dim; d++) {
                    temp_position[d] = (double)neighbors[i * dim + d];
                }
                neighbor_fitness[i] = (float)objective_function(temp_position);
            }
            err = clEnqueueWriteBuffer(queue, neighbor_fitness_buffer, CL_TRUE, 0, 
                                       population_size * sizeof(float), neighbor_fitness, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing neighbor fitness buffer: %d\n", err);
                goto cleanup;
            }

            // Update population
            err = clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
            err |= clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &opt->fitness_buffer);
            err |= clSetKernelArg(update_kernel, 2, sizeof(cl_mem), &neighbor_buffer);
            err |= clSetKernelArg(update_kernel, 3, sizeof(cl_mem), &neighbor_fitness_buffer);
            err |= clSetKernelArg(update_kernel, 4, sizeof(int), &dim);
            err |= clSetKernelArg(update_kernel, 5, sizeof(int), &population_size);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error setting update kernel args: %d\n", err);
                goto cleanup;
            }

            err = clEnqueueNDRangeKernel(queue, update_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error enqueuing update kernel: %d\n", err);
                goto cleanup;
            }

            // Re-evaluate population fitness on host
            err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, 
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
            err = clEnqueueWriteBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, 
                                       population_size * sizeof(float), fitness, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing fitness buffer: %d\n", err);
                goto cleanup;
            }
        }

        // Find best solution
        err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &best_idx_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(int), &population_size);
        err |= clSetKernelArg(best_kernel, 3, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 4, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best kernel args: %d\n", err);
            goto cleanup;
        }

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
        float best_fitness;
        err = clEnqueueReadBuffer(queue, opt->fitness_buffer, CL_TRUE, best_idx * sizeof(float), sizeof(float), &best_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best fitness: %d\n", err);
            goto cleanup;
        }
        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)best_fitness;
            float* best_solution = (float*)malloc(dim * sizeof(float));
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
                opt->best_solution.position[d] = (double)best_solution[d];
            }
            free(best_solution);
        }

        // Profiling
        cl_ulong time_start, time_end;
        double init_time = 0, neighbor_time = 0, update_time = 0, best_time = 0;
        cl_ulong queue_properties;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(cl_ulong), &queue_properties, NULL);
        if (err == CL_SUCCESS && (queue_properties & CL_QUEUE_PROFILING_ENABLE)) {
            if (events[0]) {
                err = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                    if (err == CL_SUCCESS) init_time = (time_end - time_start) / 1e6;
                }
            }
            if (events[1]) {
                err = clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                    if (err == CL_SUCCESS) neighbor_time = (time_end - time_start) / 1e6;
                }
            }
            if (events[2]) {
                err = clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                    if (err == CL_SUCCESS) update_time = (time_end - time_start) / 1e6;
                }
            }
            if (events[3]) {
                err = clGetEventProfilingInfo(events[3], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(events[3], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                    if (err == CL_SUCCESS) best_time = (time_end - time_start) / 1e6;
                }
            }
            end_time = get_time_ms();
            printf("VNS|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Neighbor: %.3f ms | Update: %.3f ms | Best: %.3f ms\n", 
                   iter + 1, opt->best_solution.fitness, end_time - start_time, 
                   init_time, neighbor_time, update_time, best_time);
        } else {
            end_time = get_time_ms();
            printf("VNS|%5d -----> %9.16f | Total: %.3f ms | Profiling disabled\n", 
                   iter + 1, opt->best_solution.fitness, end_time - start_time);
        }
    }

    // Update CPU-side population
    err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, 
                              population_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final population buffer: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: Population[%d].position is null during final update\n", i);
            goto cleanup;
        }
        for (int d = 0; d < dim; d++) {
            opt->population[i].position[d] = (double)population[i * dim + d];
        }
        opt->population[i].fitness = (double)fitness[i];
    }

cleanup:
    if (population) free(population);
    if (fitness) free(fitness);
    if (neighbors) free(neighbors);
    if (neighbor_fitness) free(neighbor_fitness);
    if (best_idx_array) free(best_idx_array);
    if (bounds_float) free(bounds_float);
    if (temp_position) free(temp_position);

    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (neighbor_buffer) clReleaseMemObject(neighbor_buffer);
    if (neighbor_fitness_buffer) clReleaseMemObject(neighbor_fitness_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (opt->population_buffer) clReleaseMemObject(opt->population_buffer);
    if (opt->fitness_buffer) clReleaseMemObject(opt->fitness_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (neighbor_kernel) clReleaseKernel(neighbor_kernel);
    if (update_kernel) clReleaseKernel(update_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 4; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (queue) clFinish(queue);
}
