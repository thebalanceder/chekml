/* EA.c - GPU-Optimized Evolutionary Algorithm with Flexible Objective Function */
#include "EA.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h>

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

// OpenCL kernel source (no hardcoded objective function)
static const char* ea_kernel_source =
"#define EA_MU 5.0f\n"
"#define EA_MUM 10.0f\n"
"#define EA_CROSSOVER_PROB 0.9f\n"
"#define EA_MUTATION_PROB 0.3f\n"
"#define EA_TOURNAMENT_SIZE 4\n"
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
"    __global float* best_solution,\n"
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
"        if (id == 0) {\n"
"            for (int d = 0; d < dim; d++) best_solution[d] = 0.0f;\n"
"        }\n"
"    }\n"
"}\n"
// Evolve: Tournament selection, SBX crossover, mutation
"__kernel void evolve(\n"
"    __global float* population,\n"
"    __global float* fitness,\n"
"    __global float* best_solution,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < population_size / 2) {\n"
"        uint local_seed = seed + id * dim;\n"
"        int idx1 = id * 2;\n"
"        int idx2 = id * 2 + 1;\n"
"        // Tournament selection\n"
"        int p1 = 0, p2 = 0;\n"
"        float p1_fitness = INFINITY, p2_fitness = INFINITY;\n"
"        for (int t = 0; t < EA_TOURNAMENT_SIZE; t++) {\n"
"            int idx = lcg_rand(&local_seed) % population_size;\n"
"            if (fitness[idx] < p1_fitness) {\n"
"                p1_fitness = fitness[idx];\n"
"                p1 = idx;\n"
"            }\n"
"            idx = lcg_rand(&local_seed) % population_size;\n"
"            if (fitness[idx] < p2_fitness) {\n"
"                p2_fitness = fitness[idx];\n"
"                p2 = idx;\n"
"            }\n"
"        }\n"
"        // Load bounds to local memory\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        // SBX crossover\n"
"        float offspring1[10], offspring2[10]; // Fixed size for dim=10\n"
"        if (lcg_rand_float(&local_seed, 0.0f, 1.0f) < EA_CROSSOVER_PROB) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float x1 = population[p1 * dim + d];\n"
"                float x2 = population[p2 * dim + d];\n"
"                float u = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"                float beta = (u <= 0.5f) ? pow(2.0f * u, 1.0f / (EA_MU + 1.0f)) :\n"
"                             pow(1.0f / (2.0f * (1.0f - u)), 1.0f / (EA_MU + 1.0f));\n"
"                float min = local_bounds[2 * d];\n"
"                float max = local_bounds[2 * d + 1];\n"
"                float c1 = 0.5f * ((1.0f + beta) * x1 + (1.0f - beta) * x2);\n"
"                float c2 = 0.5f * ((1.0f - beta) * x1 + (1.0f + beta) * x2);\n"
"                offspring1[d] = clamp(c1, min, max);\n"
"                offspring2[d] = clamp(c2, min, max);\n"
"            }\n"
"        } else {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                offspring1[d] = population[p1 * dim + d];\n"
"                offspring2[d] = population[p2 * dim + d];\n"
"            }\n"
"        }\n"
"        // Polynomial mutation\n"
"        for (int d = 0; d < dim; d++) {\n"
"            if (lcg_rand_float(&local_seed, 0.0f, 1.0f) < EA_MUTATION_PROB) {\n"
"                float r = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"                float delta = (r < 0.5f) ? pow(2.0f * r, 1.0f / (EA_MUM + 1.0f)) - 1.0f :\n"
"                              1.0f - pow(2.0f * (1.0f - r), 1.0f / (EA_MUM + 1.0f));\n"
"                float min = local_bounds[2 * d];\n"
"                float max = local_bounds[2 * d + 1];\n"
"                offspring1[d] = clamp(offspring1[d] + delta, min, max);\n"
"            }\n"
"            if (lcg_rand_float(&local_seed, 0.0f, 1.0f) < EA_MUTATION_PROB) {\n"
"                float r = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"                float delta = (r < 0.5f) ? pow(2.0f * r, 1.0f / (EA_MUM + 1.0f)) - 1.0f :\n"
"                              1.0f - pow(2.0f * (1.0f - r), 1.0f / (EA_MUM + 1.0f));\n"
"                float min = local_bounds[2 * d];\n"
"                float max = local_bounds[2 * d + 1];\n"
"                offspring2[d] = clamp(offspring2[d] + delta, min, max);\n"
"            }\n"
"        }\n"
"        // Update population\n"
"        for (int d = 0; d < dim; d++) {\n"
"            population[idx1 * dim + d] = offspring1[d];\n"
"            population[idx2 * dim + d] = offspring2[d];\n"
"        }\n"
"    }\n"
"}\n";

void EA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, evolve_kernel = NULL;
    cl_mem bounds_buffer = NULL, best_solution_buffer = NULL, best_fitness_buffer = NULL;
    float* bounds_float = NULL;
    float* population = NULL;
    float* fitness = NULL;
    double* temp_position = NULL;
    cl_event events[3] = {0};
    double start_time, end_time, transfer_time, fitness_time;
    cl_context new_context = NULL;
    cl_command_queue new_queue = NULL;

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
        if (err != CL_SUCCESS || num_devices == 0) {
            printf("No GPU devices or error (%d) for platform %u\n", err, i);
            continue;
        }
        printf("Found %u GPU devices for platform %u\n", num_devices, i);
        if (num_devices > 1000) {
            fprintf(stderr, "Error: Unreasonable number of devices (%u) for platform %u\n", num_devices, i);
            continue;
        }
        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        if (!devices) {
            fprintf(stderr, "Error: Memory allocation failed for devices\n");
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
            if (strstr(platform_name, "Intel(R) OpenCL HD Graphics") &&
                strstr(device_name, "Intel(R) Graphics [0x46a8]")) {
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
    cl_queue_properties queue_props[] = {
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
        0
    };
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
    program = clCreateProgramWithSource(new_context, 1, &ea_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating EA program: %d\n", err);
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
            fprintf(stderr, "Error building EA program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(new_queue);
        clReleaseContext(new_context);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    evolve_kernel = clCreateKernel(program, "evolve", &err);
    if (err != CL_SUCCESS || !init_kernel || !evolve_kernel) {
        fprintf(stderr, "Error creating EA kernels: %d\n", err);
        goto cleanup;
    }

    // Query kernel work-group size
    size_t init_preferred_multiple, evolve_preferred_multiple;
    err = clGetKernelWorkGroupInfo(init_kernel, gpu_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                   sizeof(size_t), &init_preferred_multiple, NULL);
    err |= clGetKernelWorkGroupInfo(evolve_kernel, gpu_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                    sizeof(size_t), &evolve_preferred_multiple, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Warning: Error querying kernel preferred work-group size: %d, using default %zu\n", err, local_work_size);
    } else {
        local_work_size = init_preferred_multiple < local_work_size ? init_preferred_multiple : local_work_size;
        printf("Preferred work-group size multiple: init=%zu, evolve=%zu\n",
               init_preferred_multiple, evolve_preferred_multiple);
    }

    // Create buffers
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    population = (float*)malloc(population_size * dim * sizeof(float));
    fitness = (float*)malloc(population_size * sizeof(float));
    temp_position = (double*)malloc(dim * sizeof(double));
    if (!bounds_float || !population || !fitness || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }

    bounds_buffer = clCreateBuffer(new_context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    best_solution_buffer = clCreateBuffer(new_context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    best_fitness_buffer = clCreateBuffer(new_context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
    opt->population_buffer = clCreateBuffer(new_context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    opt->fitness_buffer = clCreateBuffer(new_context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !best_solution_buffer || !best_fitness_buffer || !opt->population_buffer || !opt->fitness_buffer) {
        fprintf(stderr, "Error creating EA buffers: %d\n", err);
        goto cleanup;
    }

    // Write bounds asynchronously
    transfer_time = get_time_ms();
    err = clEnqueueWriteBuffer(new_queue, bounds_buffer, CL_FALSE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, &events[0]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bounds buffer: %d\n", err);
        goto cleanup;
    }
    clWaitForEvents(1, &events[0]);
    transfer_time = get_time_ms() - transfer_time;
    printf("Bounds transfer time: %.3f ms\n", transfer_time);

    // Initialize population
    uint seed = (uint)time(NULL);
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &best_solution_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 3, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 4, sizeof(int), &population_size);
    err |= clSetKernelArg(init_kernel, 5, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting init kernel args: %d\n", err);
        goto cleanup;
    }

    size_t init_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(new_queue, init_kernel, 1, NULL, &init_global_work_size, &local_work_size, 1, &events[0], &events[1]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel: %d\n", err);
        goto cleanup;
    }

    // Compute initial fitness on CPU
    transfer_time = get_time_ms();
    err = clEnqueueReadBuffer(new_queue, opt->population_buffer, CL_FALSE, 0,
                              population_size * dim * sizeof(float), population, 1, &events[1], &events[0]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading initial population: %d\n", err);
        goto cleanup;
    }
    clWaitForEvents(1, &events[0]);
    transfer_time = get_time_ms() - transfer_time;

    fitness_time = get_time_ms();
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            temp_position[d] = (double)population[i * dim + d];
        }
        fitness[i] = (float)objective_function(temp_position);
        if (fitness[i] < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness[i];
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = temp_position[d];
            }
        }
    }
    fitness_time = get_time_ms() - fitness_time;

    // Write initial fitness to GPU
    err = clEnqueueWriteBuffer(new_queue, opt->fitness_buffer, CL_FALSE, 0,
                               population_size * sizeof(float), fitness, 0, NULL, &events[1]);
    err |= clEnqueueWriteBuffer(new_queue, best_fitness_buffer, CL_FALSE, 0,
                                sizeof(float), &fitness[0], 0, NULL, &events[2]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial fitness buffers: %d\n", err);
        goto cleanup;
    }
    clWaitForEvents(2, &events[1]);

    printf("Initial population transfer time: %.3f ms, Fitness computation: %.3f ms\n", transfer_time, fitness_time);

    // Main loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Evolve
        double setup_time = get_time_ms();
        err = clSetKernelArg(evolve_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(evolve_kernel, 1, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(evolve_kernel, 2, sizeof(cl_mem), &best_solution_buffer);
        err |= clSetKernelArg(evolve_kernel, 3, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(evolve_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(evolve_kernel, 5, sizeof(int), &population_size);
        err |= clSetKernelArg(evolve_kernel, 6, sizeof(uint), &seed);
        err |= clSetKernelArg(evolve_kernel, 7, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting evolve kernel args: %d\n", err);
            goto cleanup;
        }

        size_t evolve_global_work_size = ((population_size / 2 + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(new_queue, evolve_kernel, 1, NULL, &evolve_global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing evolve kernel: %d\n", err);
            goto cleanup;
        }

        // Read population and compute fitness on CPU
        transfer_time = get_time_ms();
        err = clEnqueueReadBuffer(new_queue, opt->population_buffer, CL_FALSE, 0,
                                  population_size * dim * sizeof(float), population, 1, &events[2], &events[0]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading population: %d\n", err);
            goto cleanup;
        }
        clWaitForEvents(1, &events[0]);
        transfer_time = get_time_ms() - transfer_time;

        fitness_time = get_time_ms();
        for (int i = 0; i < population_size; i++) {
            for (int d = 0; d < dim; d++) {
                temp_position[d] = (double)population[i * dim + d];
            }
            fitness[i] = (float)objective_function(temp_position);
            if (fitness[i] < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness[i];
                for (int d = 0; d < dim; d++) {
                    opt->best_solution.position[d] = temp_position[d];
                }
            }
        }
        fitness_time = get_time_ms() - fitness_time;

        // Write fitness back to GPU
        err = clEnqueueWriteBuffer(new_queue, opt->fitness_buffer, CL_FALSE, 0,
                                   population_size * sizeof(float), fitness, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            goto cleanup;
        }
        clWaitForEvents(1, &events[1]);
        setup_time = get_time_ms() - setup_time;

        // Log every 10 iterations or last iteration
        if ((iter + 1) % 10 == 0 || iter == max_iter - 1) {
            cl_ulong time_start, time_end;
            double init_time = 0, evolve_time = 0;
            cl_ulong queue_properties;
            err = clGetCommandQueueInfo(new_queue, CL_QUEUE_PROPERTIES, sizeof(cl_ulong), &queue_properties, NULL);
            if (err == CL_SUCCESS && (queue_properties & CL_QUEUE_PROFILING_ENABLE)) {
                if (events[1]) {
                    err = clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                    if (err == CL_SUCCESS) {
                        err = clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                        if (err == CL_SUCCESS) init_time = (time_end - time_start) / 1e6;
                    }
                }
                if (events[2]) {
                    err = clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                    if (err == CL_SUCCESS) {
                        err = clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                        if (err == CL_SUCCESS) evolve_time = (time_end - time_start) / 1e6;
                    }
                }
                end_time = get_time_ms();
                printf("EA|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Evolve: %.3f ms | Setup: %.3f ms | Transfer: %.3f ms | Fitness: %.3f ms\n",
                       iter + 1, opt->best_solution.fitness, end_time - start_time,
                       init_time, evolve_time, setup_time, transfer_time, fitness_time);
            } else {
                end_time = get_time_ms();
                printf("EA|%5d -----> %9.16f | Total: %.3f ms | Profiling disabled | Setup: %.3f ms | Transfer: %.3f ms | Fitness: %.3f ms\n",
                       iter + 1, opt->best_solution.fitness, end_time - start_time, setup_time, transfer_time, fitness_time);
            }
        }
    }

    // Update CPU-side population and best solution
    transfer_time = get_time_ms();
    err = clEnqueueReadBuffer(new_queue, opt->population_buffer, CL_FALSE, 0,
                              population_size * dim * sizeof(float), population, 0, NULL, &events[0]);
    err |= clEnqueueReadBuffer(new_queue, best_solution_buffer, CL_FALSE, 0,
                               dim * sizeof(float), population, 0, NULL, &events[1]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final buffers: %d\n", err);
        goto cleanup;
    }
    clWaitForEvents(2, events);
    transfer_time = get_time_ms() - transfer_time;

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
    printf("Final population transfer time: %.3f ms\n", transfer_time);

cleanup:
    if (population) free(population);
    if (fitness) free(fitness);
    if (temp_position) free(temp_position);
    if (bounds_float) free(bounds_float);

    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (best_solution_buffer) clReleaseMemObject(best_solution_buffer);
    if (best_fitness_buffer) clReleaseMemObject(best_fitness_buffer);
    if (opt->population_buffer) clReleaseMemObject(opt->population_buffer);
    if (opt->fitness_buffer) clReleaseMemObject(opt->fitness_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (evolve_kernel) clReleaseKernel(evolve_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 3; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (new_queue) clReleaseCommandQueue(new_queue);
    if (new_context) clReleaseContext(new_context);
    if (new_queue) clFinish(new_queue);
}

void EA_free(Optimizer* opt) {
    if (opt->population) {
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].position) {
                free(opt->population[i].position);
                opt->population[i].position = NULL;
            }
        }
        free(opt->population);
        opt->population = NULL;
    }
    if (opt->best_solution.position) {
        free(opt->best_solution.position);
        opt->best_solution.position = NULL;
    }
}
