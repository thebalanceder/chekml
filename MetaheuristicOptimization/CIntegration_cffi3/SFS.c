/* SFS.c - GPU-Optimized Stochastic Fractal Search */
#include "SFS.h"
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
static const char* sfs_kernel_source =
"#define SFS_WALK_PROB 0.8f\n"
"#define SFS_MAX_DIFFUSION 4\n"
"#define SFS_EARLY_STOP_THRESHOLD 1e-6f\n"
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
"float gaussian_rand(uint* seed, float mean, float sigma) {\n"
"    float u1 = lcg_rand_float(seed, 0.0f, 1.0f);\n"
"    float u2 = lcg_rand_float(seed, 0.0f, 1.0f);\n"
"    return mean + sigma * sqrt(-2.0f * log(u1)) * cos(2.0f * 3.14159265359f * u2);\n"
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
"__kernel void diffusion_process(\n"
"    __global const float* population,\n"
"    __global float* new_points,\n"
"    __global const float* best_point,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int generation,\n"
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
"        float distance = 0.0f;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float diff = population[id * dim + d] - best_point[d];\n"
"            distance += diff * diff;\n"
"        }\n"
"        distance = sqrt(distance);\n"
"        float sigma = (log((float)(generation + 1)) / (float)(generation + 1)) * distance;\n"
"        for (int i = 0; i < SFS_MAX_DIFFUSION; i++) {\n"
"            int point_idx = id * SFS_MAX_DIFFUSION + i;\n"
"            if (lcg_rand_float(&local_seed, 0.0f, 1.0f) < SFS_WALK_PROB) {\n"
"                for (int d = 0; d < dim; d++) {\n"
"                    float term1 = gaussian_rand(&local_seed, best_point[d], sigma);\n"
"                    float term2 = lcg_rand_float(&local_seed, -1.0f, 1.0f) * best_point[d];\n"
"                    float term3 = lcg_rand_float(&local_seed, -1.0f, 1.0f) * population[id * dim + d];\n"
"                    float new_pos = term1 + (term2 - term3);\n"
"                    new_pos = fmax(local_bounds[2 * d], fmin(local_bounds[2 * d + 1], new_pos));\n"
"                    new_points[point_idx * dim + d] = new_pos;\n"
"                }\n"
"            } else {\n"
"                for (int d = 0; d < dim; d++) {\n"
"                    float new_pos = gaussian_rand(&local_seed, population[id * dim + d], sigma);\n"
"                    new_pos = fmax(local_bounds[2 * d], fmin(local_bounds[2 * d + 1], new_pos));\n"
"                    new_points[point_idx * dim + d] = new_pos;\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void select_best_diffusion(\n"
"    __global const float* new_points,\n"
"    __global const float* new_fitness,\n"
"    __global float* population,\n"
"    __global float* fitness,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    __local float* local_fitness,\n"
"    __local int* local_indices)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < population_size) {\n"
"        int base_idx = id * SFS_MAX_DIFFUSION;\n"
"        local_fitness[local_id] = new_fitness[base_idx];\n"
"        local_indices[local_id] = base_idx;\n"
"        for (int i = 1; i < SFS_MAX_DIFFUSION; i++) {\n"
"            int idx = base_idx + i;\n"
"            if (new_fitness[idx] < local_fitness[local_id]) {\n"
"                local_fitness[local_id] = new_fitness[idx];\n"
"                local_indices[local_id] = idx;\n"
"            }\n"
"        }\n"
"        int best_idx = local_indices[local_id];\n"
"        for (int d = 0; d < dim; d++) {\n"
"            population[id * dim + d] = new_points[best_idx * dim + d];\n"
"        }\n"
"        fitness[id] = local_fitness[local_id];\n"
"    }\n"
"}\n"
"__kernel void first_update_process(\n"
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
"        float Pa = ((float)(population_size - id) + 1.0f) / population_size;\n"
"        int rand1 = lcg_rand(&local_seed) % population_size;\n"
"        int rand2 = lcg_rand(&local_seed) % population_size;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            if (lcg_rand_float(&local_seed, 0.0f, 1.0f) > Pa) {\n"
"                float new_pos = population[rand1 * dim + d] - \n"
"                                lcg_rand_float(&local_seed, 0.0f, 1.0f) * \n"
"                                (population[rand2 * dim + d] - population[id * dim + d]);\n"
"                new_pos = fmax(local_bounds[2 * d], fmin(local_bounds[2 * d + 1], new_pos));\n"
"                population[id * dim + d] = new_pos;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void second_update_process(\n"
"    __global float* population,\n"
"    __global const float* best_point,\n"
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
"        float Pa = ((float)(population_size - id)) / population_size;\n"
"        if (lcg_rand_float(&local_seed, 0.0f, 1.0f) > Pa) {\n"
"            int r1 = lcg_rand(&local_seed) % population_size;\n"
"            int r2 = lcg_rand(&local_seed) % population_size;\n"
"            while (r2 == r1) r2 = lcg_rand(&local_seed) % population_size;\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float new_pos;\n"
"                if (lcg_rand_float(&local_seed, 0.0f, 1.0f) < 0.5f) {\n"
"                    new_pos = population[id * dim + d] - \n"
"                              lcg_rand_float(&local_seed, 0.0f, 1.0f) * \n"
"                              (population[r2 * dim + d] - best_point[d]);\n"
"                } else {\n"
"                    new_pos = population[id * dim + d] + \n"
"                              lcg_rand_float(&local_seed, 0.0f, 1.0f) * \n"
"                              (population[r2 * dim + d] - population[r1 * dim + d]);\n"
"                }\n"
"                new_pos = fmax(local_bounds[2 * d], fmin(local_bounds[2 * d + 1], new_pos));\n"
"                population[id * dim + d] = new_pos;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void evaluate_fitness(\n"
"    __global const float* population,\n"
"    __global float* fitness,\n"
"    const int dim,\n"
"    const int population_size)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        float sum = 0.0f;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float x = population[id * dim + d];\n"
"            sum += x * x;\n"
"        }\n"
"        fitness[id] = sum;\n"
"    }\n"
"}\n"
"__kernel void find_best(\n"
"    __global const float* fitness,\n"
"    __global float* best_point,\n"
"    __global float* best_fitness,\n"
"    __global const float* population,\n"
"    const int population_size,\n"
"    const int dim,\n"
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
"        int best_idx = local_indices[0];\n"
"        best_fitness[0] = local_fitness[0];\n"
"        for (int d = 0; d < dim; d++) {\n"
"            best_point[d] = population[best_idx * dim + d];\n"
"        }\n"
"    }\n"
"}\n";

void SFS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, diffusion_kernel = NULL, select_best_kernel = NULL, first_update_kernel = NULL, second_update_kernel = NULL, fitness_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, new_points_buffer = NULL, new_fitness_buffer = NULL, best_point_buffer = NULL, best_fitness_buffer = NULL;
    float* bounds_float = NULL, *population = NULL, *fitness = NULL, *new_points = NULL, *new_fitness = NULL, *best_point = NULL;
    cl_event events[6] = {0};
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
    const int total_diffusion_points = population_size * SFS_MAX_DIFFUSION;
    if (dim < 1 || population_size < 1 || max_iter < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), or max_iter (%d)\n", 
                dim, population_size, max_iter);
        exit(EXIT_FAILURE);
    }
    if (dim > SFS_MAX_DIM) {
        fprintf(stderr, "Error: dim (%d) exceeds maximum supported value of %d\n", dim, SFS_MAX_DIM);
        exit(EXIT_FAILURE);
    }

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
        if (err == CL_SUCCESS && num_devices > 0) {
            cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
            if (!devices) {
                fprintf(stderr, "Error: Memory allocation failed for devices\n");
                continue;
            }
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
            if (err == CL_SUCCESS) {
                platform = platforms[i];
                device = devices[0]; // Select first GPU device
                free(devices);
                break;
            }
            free(devices);
        }
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
    program = clCreateProgramWithSource(context, 1, &sfs_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating SFS program: %d\n", err);
        goto cleanup;
    }
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building SFS program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        goto cleanup;
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    diffusion_kernel = clCreateKernel(program, "diffusion_process", &err);
    select_best_kernel = clCreateKernel(program, "select_best_diffusion", &err);
    first_update_kernel = clCreateKernel(program, "first_update_process", &err);
    second_update_kernel = clCreateKernel(program, "second_update_process", &err);
    fitness_kernel = clCreateKernel(program, "evaluate_fitness", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !diffusion_kernel || !select_best_kernel || 
        !first_update_kernel || !second_update_kernel || !fitness_kernel || !best_kernel) {
        fprintf(stderr, "Error creating SFS kernels: %d\n", err);
        goto cleanup;
    }

    // Allocate host memory
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    population = (float*)malloc(population_size * dim * sizeof(float));
    fitness = (float*)malloc(population_size * sizeof(float));
    new_points = (float*)malloc(total_diffusion_points * dim * sizeof(float));
    new_fitness = (float*)malloc(total_diffusion_points * sizeof(float));
    best_point = (float*)malloc(dim * sizeof(float));
    temp_position = (double*)malloc(dim * sizeof(double));
    if (!bounds_float || !population || !fitness || !new_points || !new_fitness || !best_point || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }

    // Create buffers
    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    opt->population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    opt->fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    new_points_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, total_diffusion_points * dim * sizeof(float), NULL, &err);
    new_fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, total_diffusion_points * sizeof(float), NULL, &err);
    best_point_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    best_fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !opt->population_buffer || !opt->fitness_buffer || 
        !new_points_buffer || !new_fitness_buffer || !best_point_buffer || !best_fitness_buffer) {
        fprintf(stderr, "Error creating SFS buffers: %d\n", err);
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

    // Initialize best point
    err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &opt->fitness_buffer);
    err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &best_point_buffer);
    err |= clSetKernelArg(best_kernel, 2, sizeof(cl_mem), &best_fitness_buffer);
    err |= clSetKernelArg(best_kernel, 3, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(best_kernel, 4, sizeof(int), &population_size);
    err |= clSetKernelArg(best_kernel, 5, sizeof(int), &dim);
    err |= clSetKernelArg(best_kernel, 6, local_work_size * sizeof(float), NULL);
    err |= clSetKernelArg(best_kernel, 7, local_work_size * sizeof(int), NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting best kernel args: %d\n", err);
        goto cleanup;
    }
    err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing best kernel: %d\n", err);
        goto cleanup;
    }

    // Main loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Diffusion process
        err = clSetKernelArg(diffusion_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(diffusion_kernel, 1, sizeof(cl_mem), &new_points_buffer);
        err |= clSetKernelArg(diffusion_kernel, 2, sizeof(cl_mem), &best_point_buffer);
        err |= clSetKernelArg(diffusion_kernel, 3, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(diffusion_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(diffusion_kernel, 5, sizeof(int), &population_size);
        err |= clSetKernelArg(diffusion_kernel, 6, sizeof(int), &iter);
        err |= clSetKernelArg(diffusion_kernel, 7, sizeof(uint), &seed);
        err |= clSetKernelArg(diffusion_kernel, 8, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting diffusion kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, diffusion_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing diffusion kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate new points fitness on host
        err = clEnqueueReadBuffer(queue, new_points_buffer, CL_TRUE, 0, total_diffusion_points * dim * sizeof(float), new_points, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading new points buffer: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < total_diffusion_points; i++) {
            for (int d = 0; d < dim; d++) {
                temp_position[d] = (double)new_points[i * dim + d];
            }
            new_fitness[i] = (float)objective_function(temp_position);
        }
        err = clEnqueueWriteBuffer(queue, new_fitness_buffer, CL_TRUE, 0, total_diffusion_points * sizeof(float), new_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing new fitness buffer: %d\n", err);
            goto cleanup;
        }

        // Select best diffusion points
        err = clSetKernelArg(select_best_kernel, 0, sizeof(cl_mem), &new_points_buffer);
        err |= clSetKernelArg(select_best_kernel, 1, sizeof(cl_mem), &new_fitness_buffer);
        err |= clSetKernelArg(select_best_kernel, 2, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(select_best_kernel, 3, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(select_best_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(select_best_kernel, 5, sizeof(int), &population_size);
        err |= clSetKernelArg(select_best_kernel, 6, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(select_best_kernel, 7, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting select best kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, select_best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing select best kernel: %d\n", err);
            goto cleanup;
        }

        // First update process
        err = clSetKernelArg(first_update_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(first_update_kernel, 1, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(first_update_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(first_update_kernel, 3, sizeof(int), &population_size);
        err |= clSetKernelArg(first_update_kernel, 4, sizeof(uint), &seed);
        err |= clSetKernelArg(first_update_kernel, 5, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting first update kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, first_update_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[3]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing first update kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate fitness after first update on host
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

        // Second update process
        err = clSetKernelArg(second_update_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(second_update_kernel, 1, sizeof(cl_mem), &best_point_buffer);
        err |= clSetKernelArg(second_update_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(second_update_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(second_update_kernel, 4, sizeof(int), &population_size);
        err |= clSetKernelArg(second_update_kernel, 5, sizeof(uint), &seed);
        err |= clSetKernelArg(second_update_kernel, 6, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting second update kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, second_update_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[4]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing second update kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate fitness after second update on host
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
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &best_point_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(cl_mem), &best_fitness_buffer);
        err |= clSetKernelArg(best_kernel, 3, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(best_kernel, 4, sizeof(int), &population_size);
        err |= clSetKernelArg(best_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(best_kernel, 6, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 7, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[5]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing best kernel: %d\n", err);
            goto cleanup;
        }

        // Update best solution on host
        float best_fitness;
        err = clEnqueueReadBuffer(queue, best_fitness_buffer, CL_TRUE, 0, sizeof(float), &best_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best fitness: %d\n", err);
            goto cleanup;
        }
        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)best_fitness;
            err = clEnqueueReadBuffer(queue, best_point_buffer, CL_TRUE, 0, dim * sizeof(float), best_point, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best point: %d\n", err);
                goto cleanup;
            }
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)best_point[d];
            }
        }

        // Profiling
        cl_ulong time_start, time_end;
        double init_time = 0, diffusion_time = 0, select_time = 0, first_update_time = 0, second_update_time = 0, best_time = 0;
        if (events[0]) {
            clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            init_time = (time_end - time_start) / 1e6;
        }
        if (events[1]) {
            clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            diffusion_time = (time_end - time_start) / 1e6;
        }
        if (events[2]) {
            clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            select_time = (time_end - time_start) / 1e6;
        }
        if (events[3]) {
            clGetEventProfilingInfo(events[3], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[3], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            first_update_time = (time_end - time_start) / 1e6;
        }
        if (events[4]) {
            clGetEventProfilingInfo(events[4], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[4], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            second_update_time = (time_end - time_start) / 1e6;
        }
        if (events[5]) {
            clGetEventProfilingInfo(events[5], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            clGetEventProfilingInfo(events[5], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            best_time = (time_end - time_start) / 1e6;
        }
        end_time = get_time_ms();
        printf("SFS|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Diffusion: %.3f ms | Select: %.3f ms | First: %.3f ms | Second: %.3f ms | Best: %.3f ms\n", 
               iter + 1, opt->best_solution.fitness, end_time - start_time, init_time, diffusion_time, select_time, first_update_time, second_update_time, best_time);
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
        opt->population[i].fitness = (double)fitness[i];
    }

cleanup:
    if (bounds_float) free(bounds_float);
    if (population) free(population);
    if (fitness) free(fitness);
    if (new_points) free(new_points);
    if (new_fitness) free(new_fitness);
    if (best_point) free(best_point);
    if (temp_position) free(temp_position);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (opt->population_buffer) clReleaseMemObject(opt->population_buffer);
    if (opt->fitness_buffer) clReleaseMemObject(opt->fitness_buffer);
    if (new_points_buffer) clReleaseMemObject(new_points_buffer);
    if (new_fitness_buffer) clReleaseMemObject(new_fitness_buffer);
    if (best_point_buffer) clReleaseMemObject(best_point_buffer);
    if (best_fitness_buffer) clReleaseMemObject(best_fitness_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (diffusion_kernel) clReleaseKernel(diffusion_kernel);
    if (select_best_kernel) clReleaseKernel(select_best_kernel);
    if (first_update_kernel) clReleaseKernel(first_update_kernel);
    if (second_update_kernel) clReleaseKernel(second_update_kernel);
    if (fitness_kernel) clReleaseKernel(fitness_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 6; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (queue) clFinish(queue);
}
