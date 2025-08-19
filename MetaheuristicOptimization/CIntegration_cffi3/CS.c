/* CS.c - GPU-Optimized Clonal Selection Optimization with Flexible Objective Function */
#include "CS.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
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
static const char* cs_kernel_source =
"#define CS_MUTATION_PROBABILITY 0.1f\n" // Add definition
"// Simple LCG random number generator\n"
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
"__kernel void clone_and_hypermutate(\n"
"    __global const float* population,\n"
"    __global float* clones,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int clone_size,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int total_clones = population_size * clone_size;\n"
"    if (id < total_clones) {\n"
"        int orig_idx = id / clone_size;\n"
"        uint local_seed = seed + id;\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float pos = population[orig_idx * dim + d];\n"
"            if (lcg_rand_float(&local_seed, 0.0f, 1.0f) < CS_MUTATION_PROBABILITY) {\n"
"                float min = local_bounds[2 * d];\n"
"                float max = local_bounds[2 * d + 1];\n"
"                pos = lcg_rand_float(&local_seed, min, max);\n"
"            }\n"
"            clones[id * dim + d] = pos;\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void update_population(\n"
"    __global float* population,\n"
"    __global float* population_fitness,\n"
"    __global const float* clones,\n"
"    __global const float* clone_fitness,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int clone_size,\n"
"    const int replace_count,\n"
"    uint seed,\n"
"    __local float* local_fitness,\n"
"    __local int* local_indices,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    if (id < population_size) {\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        int base_idx = id * clone_size;\n"
"        local_fitness[local_id] = clone_fitness[base_idx];\n"
"        local_indices[local_id] = base_idx;\n"
"        for (int offset = 1; offset < clone_size; offset++) {\n"
"            int clone_idx = base_idx + offset;\n"
"            if (clone_idx < population_size * clone_size && clone_fitness[clone_idx] < local_fitness[local_id]) {\n"
"                local_fitness[local_id] = clone_fitness[clone_idx];\n"
"                local_indices[local_id] = clone_idx;\n"
"            }\n"
"        }\n"
"        if (id < population_size - replace_count) {\n"
"            int best_clone_idx = local_indices[local_id];\n"
"            for (int d = 0; d < dim; d++) {\n"
"                population[id * dim + d] = clones[best_clone_idx * dim + d];\n"
"            }\n"
"            population_fitness[id] = local_fitness[local_id];\n"
"        } else {\n"
"            uint local_seed = seed + id;\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float min = local_bounds[2 * d];\n"
"                float max = local_bounds[2 * d + 1];\n"
"                population[id * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"            }\n"
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

void CS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, clone_kernel = NULL, update_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, clone_buffer = NULL, clone_fitness_buffer = NULL, best_idx_buffer = NULL;
    float* bounds_float = NULL, *population = NULL, *fitness = NULL, *clones = NULL, *clone_fitness = NULL;
    int* best_idx_array = NULL;
    cl_event events[4] = {0};
    double start_time, end_time;
    cl_context new_context = NULL;
    cl_command_queue new_queue = NULL;
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

    const int clone_size = (int)(CS_CLONE_FACTOR * population_size);
    const int total_clones = population_size * clone_size;
    const int replace_count = (int)(CS_REPLACEMENT_RATE * population_size);
    if (clone_size < 1 || total_clones < population_size || replace_count < 1) {
        fprintf(stderr, "Error: Invalid clone_size (%d), total_clones (%d), or replace_count (%d)\n", 
                clone_size, total_clones, replace_count);
        exit(EXIT_FAILURE);
    }

    // Select GPU platform and device
    cl_platform_id gpu_platform = NULL;
    cl_device_id gpu_device = NULL;
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
        char platform_name[128] = "Unknown";
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        printf("Checking platform %u: %s\n", i, platform_name);
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error getting device count for platform %u: %d\n", i, err);
            continue;
        }
        if (num_devices == 0) {
            printf("No GPU devices found for platform %u\n", i);
            continue;
        }
        printf("Found %u GPU devices for platform %u\n", num_devices, i);
        if (num_devices > 1000) {
            fprintf(stderr, "Error: Unreasonable number of devices (%u) for platform %u\n", num_devices, i);
            continue;
        }
        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        if (!devices) {
            fprintf(stderr, "Error: Memory allocation failed for devices (num_devices=%u, platform=%u)\n", num_devices, i);
            continue;
        }
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error getting device IDs for platform %u: %d\n", i, err);
            free(devices);
            continue;
        }
        for (cl_uint j = 0; j < num_devices; j++) {
            char device_name[128] = "Unknown";
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            printf("  Device %u: %s\n", j, device_name);
            if (strstr(device_name, "Intel(R) Graphics [0x46a8]")) {
                gpu_platform = platforms[i];
                gpu_device = devices[j];
                printf("Selected device: %s\n", device_name);
            }
        }
        free(devices);
        if (gpu_platform && gpu_device) break;
    }

    if (!gpu_platform || !gpu_device) {
        fprintf(stderr, "Error: No GPU device (Intel(R) Graphics [0x46a8]) found\n");
        free(platforms);
        exit(EXIT_FAILURE);
    }

    // Create context
    new_context = clCreateContext(NULL, 1, &gpu_device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !new_context) {
        fprintf(stderr, "Error creating OpenCL context: %d\n", err);
        free(platforms);
        exit(EXIT_FAILURE);
    }

    // Create command queue with properties
    cl_queue_properties queue_props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    new_queue = clCreateCommandQueueWithProperties(new_context, gpu_device, queue_props, &err);
    if (err != CL_SUCCESS || !new_queue) {
        fprintf(stderr, "Error creating OpenCL queue: %d\n", err);
        clReleaseContext(new_context);
        free(platforms);
        exit(EXIT_FAILURE);
    }

    // Debug platform and device
    char platform_name[128] = "Unknown", device_name[128] = "Unknown";
    clGetPlatformInfo(gpu_platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    clGetDeviceInfo(gpu_device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Selected platform: %s\n", platform_name);
    printf("Selected device: %s\n", device_name);
    free(platforms);

    // Verify GPU device
    cl_device_type device_type;
    err = clGetDeviceInfo(gpu_device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
    if (err != CL_SUCCESS || device_type != CL_DEVICE_TYPE_GPU) {
        fprintf(stderr, "Error: Selected device is not a GPU (type: %lu, err: %d)\n", device_type, err);
        clReleaseCommandQueue(new_queue);
        clReleaseContext(new_context);
        exit(EXIT_FAILURE);
    }

    // Query device capabilities
    size_t max_work_group_size;
    err = clGetDeviceInfo(gpu_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying CL_DEVICE_MAX_WORK_GROUP_SIZE: %d\n", err);
        clReleaseCommandQueue(new_queue);
        clReleaseContext(new_context);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = max_work_group_size < 64 ? max_work_group_size : 64;

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = 0.0;
    }

    // Create OpenCL program
    program = clCreateProgramWithSource(new_context, 1, &cs_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating CS program: %d\n", err);
        clReleaseCommandQueue(new_queue);
        clReleaseContext(new_context);
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 1, &gpu_device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, gpu_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, gpu_device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building CS program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(new_queue);
        clReleaseContext(new_context);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    clone_kernel = clCreateKernel(program, "clone_and_hypermutate", &err);
    update_kernel = clCreateKernel(program, "update_population", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !clone_kernel || !update_kernel || !best_kernel) {
        fprintf(stderr, "Error creating CS kernels: %d\n", err);
        goto cleanup;
    }

    // Query kernel work-group size
    size_t init_preferred_multiple, clone_preferred_multiple, update_preferred_multiple, best_preferred_multiple;
    err = clGetKernelWorkGroupInfo(init_kernel, gpu_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                   sizeof(size_t), &init_preferred_multiple, NULL);
    err |= clGetKernelWorkGroupInfo(clone_kernel, gpu_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                    sizeof(size_t), &clone_preferred_multiple, NULL);
    err |= clGetKernelWorkGroupInfo(update_kernel, gpu_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                    sizeof(size_t), &update_preferred_multiple, NULL);
    err |= clGetKernelWorkGroupInfo(best_kernel, gpu_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                    sizeof(size_t), &best_preferred_multiple, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Warning: Error querying kernel preferred work-group size: %d, using default %zu\n", err, local_work_size);
    } else {
        local_work_size = init_preferred_multiple < local_work_size ? init_preferred_multiple : local_work_size;
        printf("Preferred work-group size multiple: init=%zu, clone=%zu, update=%zu, best=%zu\n", 
               init_preferred_multiple, clone_preferred_multiple, update_preferred_multiple, best_preferred_multiple);
    }

    // Create buffers
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    population = (float*)malloc(population_size * dim * sizeof(float));
    fitness = (float*)malloc(population_size * sizeof(float));
    clones = (float*)malloc(total_clones * dim * sizeof(float));
    clone_fitness = (float*)malloc(total_clones * sizeof(float));
    best_idx_array = (int*)malloc(population_size * sizeof(int));
    temp_position = (double*)malloc(dim * sizeof(double));
    if (!bounds_float || !population || !fitness || !clones || !clone_fitness || !best_idx_array || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }

    bounds_buffer = clCreateBuffer(new_context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    clone_buffer = clCreateBuffer(new_context, CL_MEM_READ_WRITE, total_clones * dim * sizeof(float), NULL, &err);
    clone_fitness_buffer = clCreateBuffer(new_context, CL_MEM_READ_WRITE, total_clones * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(new_context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    opt->population_buffer = clCreateBuffer(new_context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    opt->fitness_buffer = clCreateBuffer(new_context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !clone_buffer || !clone_fitness_buffer || 
        !best_idx_buffer || !opt->population_buffer || !opt->fitness_buffer) {
        fprintf(stderr, "Error creating CS buffers: %d\n", err);
        goto cleanup;
    }

    // Write bounds
    err = clEnqueueWriteBuffer(new_queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
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

    size_t init_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(new_queue, init_kernel, 1, NULL, &init_global_work_size, &local_work_size, 0, NULL, &events[0]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel: %d\n", err);
        goto cleanup;
    }

    // Evaluate initial population fitness on host
    err = clEnqueueReadBuffer(new_queue, opt->population_buffer, CL_TRUE, 0, 
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
    err = clEnqueueWriteBuffer(new_queue, opt->fitness_buffer, CL_TRUE, 0, 
                               population_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing fitness buffer: %d\n", err);
        goto cleanup;
    }

    // Main loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Clone and hypermutate
        err = clSetKernelArg(clone_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(clone_kernel, 1, sizeof(cl_mem), &clone_buffer);
        err |= clSetKernelArg(clone_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(clone_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(clone_kernel, 4, sizeof(int), &population_size);
        err |= clSetKernelArg(clone_kernel, 5, sizeof(int), &clone_size);
        err |= clSetKernelArg(clone_kernel, 6, sizeof(uint), &seed);
        err |= clSetKernelArg(clone_kernel, 7, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting clone kernel args: %d\n", err);
            goto cleanup;
        }

        size_t clone_global_work_size = ((total_clones + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(new_queue, clone_kernel, 1, NULL, &clone_global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing clone kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate clone fitness on host
        err = clEnqueueReadBuffer(new_queue, clone_buffer, CL_TRUE, 0, 
                                  total_clones * dim * sizeof(float), clones, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading clone buffer: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < total_clones; i++) {
            for (int d = 0; d < dim; d++) {
                temp_position[d] = (double)clones[i * dim + d];
            }
            clone_fitness[i] = (float)objective_function(temp_position);
        }
        err = clEnqueueWriteBuffer(new_queue, clone_fitness_buffer, CL_TRUE, 0, 
                                   total_clones * sizeof(float), clone_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing clone fitness buffer: %d\n", err);
            goto cleanup;
        }

        // Update population
        err = clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(update_kernel, 2, sizeof(cl_mem), &clone_buffer);
        err |= clSetKernelArg(update_kernel, 3, sizeof(cl_mem), &clone_fitness_buffer);
        err |= clSetKernelArg(update_kernel, 4, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(update_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(update_kernel, 6, sizeof(int), &population_size);
        err |= clSetKernelArg(update_kernel, 7, sizeof(int), &clone_size);
        err |= clSetKernelArg(update_kernel, 8, sizeof(int), &replace_count);
        err |= clSetKernelArg(update_kernel, 9, sizeof(uint), &seed);
        err |= clSetKernelArg(update_kernel, 10, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(update_kernel, 11, local_work_size * sizeof(int), NULL);
        err |= clSetKernelArg(update_kernel, 12, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting update kernel args: %d\n", err);
            goto cleanup;
        }

        size_t update_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(new_queue, update_kernel, 1, NULL, &update_global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing update kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate updated population fitness on host
        err = clEnqueueReadBuffer(new_queue, opt->population_buffer, CL_TRUE, 0, 
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
        err = clEnqueueWriteBuffer(new_queue, opt->fitness_buffer, CL_TRUE, 0, 
                                   population_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            goto cleanup;
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

        size_t best_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(new_queue, best_kernel, 1, NULL, &best_global_work_size, &local_work_size, 0, NULL, &events[3]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing best kernel: %d\n", err);
            goto cleanup;
        }

        // Update best solution
        int best_idx;
        err = clEnqueueReadBuffer(new_queue, best_idx_buffer, CL_TRUE, 0, sizeof(int), &best_idx, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best_idx buffer: %d\n", err);
            goto cleanup;
        }
        float best_fitness;
        err = clEnqueueReadBuffer(new_queue, opt->fitness_buffer, CL_TRUE, best_idx * sizeof(float), sizeof(float), &best_fitness, 0, NULL, NULL);
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
            err = clEnqueueReadBuffer(new_queue, opt->population_buffer, CL_TRUE, best_idx * dim * sizeof(float), 
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
        double init_time = 0, clone_time = 0, update_time = 0, best_time = 0;
        cl_ulong queue_properties;
        err = clGetCommandQueueInfo(new_queue, CL_QUEUE_PROPERTIES, sizeof(cl_ulong), &queue_properties, NULL);
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
                    if (err == CL_SUCCESS) clone_time = (time_end - time_start) / 1e6;
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
            printf("CS|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Clone: %.3f ms | Update: %.3f ms | Best: %.3f ms\n", 
                   iter + 1, opt->best_solution.fitness, end_time - start_time, 
                   init_time, clone_time, update_time, best_time);
        } else {
            end_time = get_time_ms();
            printf("CS|%5d -----> %9.16f | Total: %.3f ms | Profiling disabled\n", 
                   iter + 1, opt->best_solution.fitness, end_time - start_time);
        }
    }

    // Update CPU-side population
    err = clEnqueueReadBuffer(new_queue, opt->population_buffer, CL_TRUE, 0, 
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
    if (clones) free(clones);
    if (clone_fitness) free(clone_fitness);
    if (best_idx_array) free(best_idx_array);
    if (bounds_float) free(bounds_float);
    if (temp_position) free(temp_position);

    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (clone_buffer) clReleaseMemObject(clone_buffer);
    if (clone_fitness_buffer) clReleaseMemObject(clone_fitness_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (opt->population_buffer) clReleaseMemObject(opt->population_buffer);
    if (opt->fitness_buffer) clReleaseMemObject(opt->fitness_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (clone_kernel) clReleaseKernel(clone_kernel);
    if (update_kernel) clReleaseKernel(update_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 4; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (new_queue) clReleaseCommandQueue(new_queue);
    if (new_context) clReleaseContext(new_context);
    if (new_queue) clFinish(new_queue);
}
