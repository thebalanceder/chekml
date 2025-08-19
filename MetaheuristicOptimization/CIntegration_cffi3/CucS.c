/* CucS.c - GPU-Optimized Cuckoo Search Optimization with Flexible Objective Function */
#include "CucS.h"
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
static const char* cucs_kernel_source =
"#define CS_PA 0.25f\n"
"#define CS_BETA 1.5f\n"
"#define CS_STEP_SCALE 0.01f\n"
"#define CS_SIGMA 0.696066f\n"
// Simple LCG random number generator
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
// Initialize population
"__kernel void initialize_nests(\n"
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
// Generate new solutions via Levy flights
"__kernel void get_cuckoos(\n"
"    __global float* population,\n"
"    __global const float* best_position,\n"
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
"            float s = population[id * dim + d];\n"
"            float u = lcg_rand_float(&local_seed, -1.0f, 1.0f) * CS_SIGMA;\n"
"            float v = lcg_rand_float(&local_seed, -1.0f, 1.0f);\n"
"            float step = v != 0.0f ? u / pow(fabs(v), 1.0f / CS_BETA) : u;\n"
"            float stepsize = CS_STEP_SCALE * step * (s - best_position[d]);\n"
"            s += stepsize * lcg_rand_float(&local_seed, -1.0f, 1.0f);\n"
"            // Enforce bounds\n"
"            float min = local_bounds[2 * d];\n"
"            float max = local_bounds[2 * d + 1];\n"
"            s = s < min ? min : (s > max ? max : s);\n"
"            population[id * dim + d] = s;\n"
"        }\n"
"    }\n"
"}\n"
// Replace some nests
"__kernel void empty_nests(\n"
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
"        uint local_seed = seed + id;\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float pos = population[id * dim + d];\n"
"            if (lcg_rand_float(&local_seed, 0.0f, 1.0f) > CS_PA) {\n"
"                // Select another nest randomly\n"
"                int other_idx = (int)(lcg_rand_float(&local_seed, 0.0f, (float)population_size));\n"
"                if (other_idx >= population_size) other_idx = population_size - 1;\n"
"                float other_pos = population[other_idx * dim + d];\n"
"                float stepsize = lcg_rand_float(&local_seed, 0.0f, 1.0f) * (pos - other_pos);\n"
"                pos += stepsize;\n"
"                // Enforce bounds\n"
"                float min = local_bounds[2 * d];\n"
"                float max = local_bounds[2 * d + 1];\n"
"                pos = pos < min ? min : (pos > max ? max : pos);\n"
"            }\n"
"            population[id * dim + d] = pos;\n"
"        }\n"
"    }\n"
"}\n"
// Find best solution
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

void CucS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, cuckoo_kernel = NULL, empty_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, best_position_buffer = NULL, best_idx_buffer = NULL;
    float* bounds_float = NULL, *population = NULL, *fitness = NULL, *best_position = NULL;
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
            gpu_platform = platforms[i];
            gpu_device = devices[j];
            printf("Selected device: %s\n", device_name);
        }
        free(devices);
        if (gpu_platform && gpu_device) break;
    }

    if (!gpu_platform || !gpu_device) {
        fprintf(stderr, "Error: No GPU device found\n");
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
    program = clCreateProgramWithSource(new_context, 1, &cucs_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating CucS program: %d\n", err);
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
            fprintf(stderr, "Error building CucS program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(new_queue);
        clReleaseContext(new_context);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_nests", &err);
    cuckoo_kernel = clCreateKernel(program, "get_cuckoos", &err);
    empty_kernel = clCreateKernel(program, "empty_nests", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !cuckoo_kernel || !empty_kernel || !best_kernel) {
        fprintf(stderr, "Error creating CucS kernels: %d\n", err);
        goto cleanup;
    }

    // Query kernel work-group size
    size_t init_preferred_multiple, cuckoo_preferred_multiple, empty_preferred_multiple, best_preferred_multiple;
    err = clGetKernelWorkGroupInfo(init_kernel, gpu_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                   sizeof(size_t), &init_preferred_multiple, NULL);
    err |= clGetKernelWorkGroupInfo(cuckoo_kernel, gpu_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                    sizeof(size_t), &cuckoo_preferred_multiple, NULL);
    err |= clGetKernelWorkGroupInfo(empty_kernel, gpu_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                    sizeof(size_t), &empty_preferred_multiple, NULL);
    err |= clGetKernelWorkGroupInfo(best_kernel, gpu_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                    sizeof(size_t), &best_preferred_multiple, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Warning: Error querying kernel preferred work-group size: %d, using default %zu\n", err, local_work_size);
    } else {
        local_work_size = init_preferred_multiple < local_work_size ? init_preferred_multiple : local_work_size;
        printf("Preferred work-group size multiple: init=%zu, cuckoo=%zu, empty=%zu, best=%zu\n", 
               init_preferred_multiple, cuckoo_preferred_multiple, empty_preferred_multiple, best_preferred_multiple);
    }

    // Create buffers
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    population = (float*)malloc(population_size * dim * sizeof(float));
    fitness = (float*)malloc(population_size * sizeof(float));
    best_position = (float*)malloc(dim * sizeof(float));
    best_idx_array = (int*)malloc(population_size * sizeof(int));
    temp_position = (double*)malloc(dim * sizeof(double));
    if (!bounds_float || !population || !fitness || !best_position || !best_idx_array || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }
    for (int i = 0; i < dim; i++) {
        best_position[i] = 0.0f;
    }

    bounds_buffer = clCreateBuffer(new_context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    best_position_buffer = clCreateBuffer(new_context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(new_context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    opt->population_buffer = clCreateBuffer(new_context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    opt->fitness_buffer = clCreateBuffer(new_context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !best_position_buffer || !best_idx_buffer || 
        !opt->population_buffer || !opt->fitness_buffer) {
        fprintf(stderr, "Error creating CucS buffers: %d\n", err);
        goto cleanup;
    }

    // Write bounds and initial best position
    err = clEnqueueWriteBuffer(new_queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(new_queue, best_position_buffer, CL_TRUE, 0, dim * sizeof(float), best_position, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing buffers: %d\n", err);
        goto cleanup;
    }

    // Initialize nests
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
    err = clEnqueueNDRangeKernel(new_queue, init_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[0]);
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

        // Find current best solution
        err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &best_idx_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(int), &population_size);
        err |= clSetKernelArg(best_kernel, 3, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 4, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best kernel args: %d\n", err);
            goto cleanup;
        }

        err = clEnqueueNDRangeKernel(new_queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[3]);
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
                best_position[d] = best_solution[d];
            }
            free(best_solution);
            // Update best position buffer
            err = clEnqueueWriteBuffer(new_queue, best_position_buffer, CL_TRUE, 0, dim * sizeof(float), best_position, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing best_position buffer: %d\n", err);
                goto cleanup;
            }
        }

        // Generate new solutions via Levy flights
        err = clSetKernelArg(cuckoo_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(cuckoo_kernel, 1, sizeof(cl_mem), &best_position_buffer);
        err |= clSetKernelArg(cuckoo_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(cuckoo_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(cuckoo_kernel, 4, sizeof(int), &population_size);
        err |= clSetKernelArg(cuckoo_kernel, 5, sizeof(uint), &seed);
        err |= clSetKernelArg(cuckoo_kernel, 6, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting cuckoo kernel args: %d\n", err);
            goto cleanup;
        }

        err = clEnqueueNDRangeKernel(new_queue, cuckoo_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing cuckoo kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate cuckoo fitness on host
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
            fprintf(stderr, "Error writing fitness buffer:watermark %d\n", err);
            goto cleanup;
        }

        // Replace some nests
        err = clSetKernelArg(empty_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(empty_kernel, 1, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(empty_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(empty_kernel, 3, sizeof(int), &population_size);
        err |= clSetKernelArg(empty_kernel, 4, sizeof(uint), &seed);
        err |= clSetKernelArg(empty_kernel, 5, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting empty kernel args: %d\n", err);
            goto cleanup;
        }

        err = clEnqueueNDRangeKernel(new_queue, empty_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing empty kernel: %d\n", err);
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

        // Profiling
        cl_ulong time_start, time_end;
        double init_time = 0, cuckoo_time = 0, empty_time = 0, best_time = 0;
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
                    if (err == CL_SUCCESS) cuckoo_time = (time_end - time_start) / 1e6;
                }
            }
            if (events[2]) {
                err = clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                    if (err == CL_SUCCESS) empty_time = (time_end - time_start) / 1e6;
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
            printf("CucS|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Cuckoo: %.3f ms | Empty: %.3f ms | Best: %.3f ms\n", 
                   iter + 1, opt->best_solution.fitness, end_time - start_time, 
                   init_time, cuckoo_time, empty_time, best_time);
        } else {
            end_time = get_time_ms();
            printf("CucS|%5d -----> %9.16f | Total: %.3f ms | Profiling disabled\n", 
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
    if (best_position) free(best_position);
    if (best_idx_array) free(best_idx_array);
    if (bounds_float) free(bounds_float);
    if (temp_position) free(temp_position);

    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (best_position_buffer) clReleaseMemObject(best_position_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (opt->population_buffer) clReleaseMemObject(opt->population_buffer);
    if (opt->fitness_buffer) clReleaseMemObject(opt->fitness_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (cuckoo_kernel) clReleaseKernel(cuckoo_kernel);
    if (empty_kernel) clReleaseKernel(empty_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 4; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (new_queue) clReleaseCommandQueue(new_queue);
    if (new_context) clReleaseContext(new_context);
    if (new_queue) clFinish(new_queue);
}
