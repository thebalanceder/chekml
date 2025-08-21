/* EHO.c - GPU-Optimized Elephant Herding Optimization with Flexible Objective Function */
#include "EHO.h"
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
static const char* eho_kernel_source =
"#define EHO_ALPHA 0.5f\n"
"#define EHO_BETA 0.1f\n"
"#define EHO_NUM_CLANS 5\n"
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
"__kernel void clan_updating(\n"
"    __global float* population,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int elephants_per_clan,\n"
"    uint seed,\n"
"    __local float* local_bounds,\n"
"    __local float* clan_center)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int clan_id = id / elephants_per_clan;\n"
"    int idx_in_clan = id % elephants_per_clan;\n"
"    if (id < population_size) {\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        if (local_id < dim) {\n"
"            clan_center[local_id] = 0.0f;\n"
"            for (int i = 0; i < elephants_per_clan; i++) {\n"
"                clan_center[local_id] += population[(clan_id * elephants_per_clan + i) * dim + local_id];\n"
"            }\n"
"            clan_center[local_id] /= elephants_per_clan;\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        uint local_seed = seed + id;\n"
"        __global float* current = &population[id * dim];\n"
"        __global float* best = &population[clan_id * elephants_per_clan * dim];\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float new_pos = current[d] + EHO_ALPHA * (best[d] - current[d]) * lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"            if (idx_in_clan == 0) {\n"
"                new_pos = current[d];\n"
"            } else if (new_pos == current[d]) {\n"
"                new_pos = EHO_BETA * clan_center[d];\n"
"            }\n"
"            new_pos = fmax(local_bounds[2 * d], fmin(local_bounds[2 * d + 1], new_pos));\n"
"            current[d] = new_pos;\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void separating_phase(\n"
"    __global float* population,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int elephants_per_clan,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int clan_id = id / elephants_per_clan;\n"
"    int idx_in_clan = id % elephants_per_clan;\n"
"    if (id < population_size) {\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        if (idx_in_clan == elephants_per_clan - 1) {\n"
"            uint local_seed = seed + id;\n"
"            for (int d = 0; d < dim; d++) {\n"
"                population[id * dim + d] = lcg_rand_float(&local_seed, local_bounds[2 * d], local_bounds[2 * d + 1]);\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void sort_and_elitism(\n"
"    __global float* population,\n"
"    __global float* fitness,\n"
"    __global float* elite_population,\n"
"    __global float* elite_fitness,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int keep,\n"
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
"        if (local_id < offset && local_id + offset < local_size) {\n"
"            if (local_fitness[local_id + offset] < local_fitness[local_id]) {\n"
"                local_fitness[local_id] = local_fitness[local_id + offset];\n"
"                local_indices[local_id] = local_indices[local_id + offset];\n"
"            }\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (local_id == 0) {\n"
"        int group_id = get_group_id(0);\n"
"        elite_fitness[group_id] = local_fitness[0];\n"
"        for (int d = 0; d < dim; d++) {\n"
"            elite_population[group_id * dim + d] = population[local_indices[0] * dim + d];\n"
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
void EHO_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL;
    cl_kernel clan_update_kernel = NULL;
    cl_kernel separate_kernel = NULL;
    cl_kernel sort_elite_kernel = NULL;
    cl_kernel best_kernel = NULL;
    cl_mem bounds_buffer = NULL;
    cl_mem elite_population_buffer = NULL;
    cl_mem elite_fitness_buffer = NULL;
    cl_mem best_idx_buffer = NULL;
    float* bounds_float = NULL;
    float* population = NULL;
    float* fitness = NULL;
    float* elite_population = NULL;
    float* elite_fitness = NULL;
    int* best_idx_array = NULL;
    cl_event events[5] = {0};
    double start_time;
    double end_time;
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
    if (population_size % EHO_NUM_CLANS != 0) {
        fprintf(stderr, "Error: Population size must be divisible by number of clans\n");
        exit(EXIT_FAILURE);
    }
    const int elephants_per_clan = population_size / EHO_NUM_CLANS;
    const int keep = EHO_KEEP;
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
        if (err != CL_SUCCESS || num_devices == 0) {
            continue;
        }
        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        if (!devices) {
            fprintf(stderr, "Error: Memory allocation failed for devices\n");
            continue;
        }
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
    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !context) {
        fprintf(stderr, "Error creating OpenCL context: %d\n", err);
        exit(EXIT_FAILURE);
    }
    // Create command queue
    cl_queue_properties queue_props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, queue_props, &err);
    if (err != CL_SUCCESS || !queue) {
        fprintf(stderr, "Error creating OpenCL queue: %d\n", err);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }
    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = 0.0;
    }
    // Create OpenCL program
    program = clCreateProgramWithSource(context, 1, &eho_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating EHO program: %d\n", err);
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
            fprintf(stderr, "Error building EHO program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }
    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    clan_update_kernel = clCreateKernel(program, "clan_updating", &err);
    separate_kernel = clCreateKernel(program, "separating_phase", &err);
    sort_elite_kernel = clCreateKernel(program, "sort_and_elitism", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !clan_update_kernel || !separate_kernel || !sort_elite_kernel || !best_kernel) {
        fprintf(stderr, "Error creating EHO kernels: %d\n", err);
        goto cleanup;
    }
    // Query device capabilities
    size_t max_work_group_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying CL_DEVICE_MAX_WORK_GROUP_SIZE: %d\n", err);
        goto cleanup;
    }
    size_t local_work_size = max_work_group_size < 64 ? max_work_group_size : 64;
    // Create buffers
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    population = (float*)malloc(population_size * dim * sizeof(float));
    fitness = (float*)malloc(population_size * sizeof(float));
    elite_population = (float*)malloc(EHO_KEEP * dim * sizeof(float));
    elite_fitness = (float*)malloc(EHO_KEEP * sizeof(float));
    best_idx_array = (int*)malloc(population_size * sizeof(int));
    temp_position = (double*)malloc(dim * sizeof(double));
    if (!bounds_float || !population || !fitness || !elite_population || !elite_fitness || !best_idx_array || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }
    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    opt->population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    opt->fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    elite_population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, EHO_KEEP * dim * sizeof(float), NULL, &err);
    elite_fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, EHO_KEEP * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !opt->population_buffer || !opt->fitness_buffer ||
        !elite_population_buffer || !elite_fitness_buffer || !best_idx_buffer) {
        fprintf(stderr, "Error creating EHO buffers: %d\n", err);
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
    // Evaluate initial population
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
    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();
        // Clan division and elitism (sorting)
        err = clSetKernelArg(sort_elite_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(sort_elite_kernel, 1, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(sort_elite_kernel, 2, sizeof(cl_mem), &elite_population_buffer);
        err |= clSetKernelArg(sort_elite_kernel, 3, sizeof(cl_mem), &elite_fitness_buffer);
        err |= clSetKernelArg(sort_elite_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(sort_elite_kernel, 5, sizeof(int), &population_size);
        err |= clSetKernelArg(sort_elite_kernel, 6, sizeof(int), &keep);
        err |= clSetKernelArg(sort_elite_kernel, 7, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(sort_elite_kernel, 8, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting sort_elite kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, sort_elite_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing sort_elite kernel: %d\n", err);
            goto cleanup;
        }
        // Clan updating
        err = clSetKernelArg(clan_update_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(clan_update_kernel, 1, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(clan_update_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(clan_update_kernel, 3, sizeof(int), &population_size);
        err |= clSetKernelArg(clan_update_kernel, 4, sizeof(int), &elephants_per_clan);
        err |= clSetKernelArg(clan_update_kernel, 5, sizeof(uint), &seed);
        err |= clSetKernelArg(clan_update_kernel, 6, 2 * dim * sizeof(float), NULL);
        err |= clSetKernelArg(clan_update_kernel, 7, dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting clan_update kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, clan_update_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing clan_update kernel: %d\n", err);
            goto cleanup;
        }
        // Separating phase
        err = clSetKernelArg(separate_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(separate_kernel, 1, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(separate_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(separate_kernel, 3, sizeof(int), &population_size);
        err |= clSetKernelArg(separate_kernel, 4, sizeof(int), &elephants_per_clan);
        err |= clSetKernelArg(separate_kernel, 5, sizeof(uint), &seed);
        err |= clSetKernelArg(separate_kernel, 6, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting separate kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, separate_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[3]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing separate kernel: %d\n", err);
            goto cleanup;
        }
        // Evaluate population
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
        err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[4]);
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
        double init_time = 0.0;
        double sort_time = 0.0;
        double update_time = 0.0;
        double separate_time = 0.0;
        double best_time = 0.0;
        if (events[0]) {
            clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            init_time = (time_end - time_start) / 1e6;
        }
        if (events[1]) {
            clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            sort_time = (time_end - time_start) / 1e6;
        }
        if (events[2]) {
            clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            update_time = (time_end - time_start) / 1e6;
        }
        if (events[3]) {
            clGetEventProfilingInfo(events[3], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[3], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            separate_time = (time_end - time_start) / 1e6;
        }
        if (events[4]) {
            clGetEventProfilingInfo(events[4], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[4], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            best_time = (time_end - time_start) / 1e6;
        }
        end_time = get_time_ms();
        printf("EHO|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Sort: %.3f ms | Update: %.3f ms | Separate: %.3f ms | Best: %.3f ms\n",
               iter + 1, opt->best_solution.fitness, end_time - start_time, init_time, sort_time, update_time, separate_time, best_time);
    }
    // Update CPU-side population
    err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
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
    if (bounds_float) free(bounds_float);
    if (population) free(population);
    if (fitness) free(fitness);
    if (elite_population) free(elite_population);
    if (elite_fitness) free(elite_fitness);
    if (best_idx_array) free(best_idx_array);
    if (temp_position) free(temp_position);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (opt->population_buffer) clReleaseMemObject(opt->population_buffer);
    if (opt->fitness_buffer) clReleaseMemObject(opt->fitness_buffer);
    if (elite_population_buffer) clReleaseMemObject(elite_population_buffer);
    if (elite_fitness_buffer) clReleaseMemObject(elite_fitness_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (clan_update_kernel) clReleaseKernel(clan_update_kernel);
    if (separate_kernel) clReleaseKernel(separate_kernel);
    if (sort_elite_kernel) clReleaseKernel(sort_elite_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 5; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}
