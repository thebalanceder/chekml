/* EPO.c - GPU-Optimized Emperor Penguin Optimization with Flexible Objective Function */
#include "EPO.h"
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
static const char* epo_kernel_source =
"#define STRATEGY_COUNT 3\n"
"#define STRATEGY_LINEAR 0\n"
"#define STRATEGY_EXPONENTIAL 1\n"
"#define STRATEGY_CHAOTIC 2\n"
"#define INITIAL_F 2.0f\n"
"#define INITIAL_L 1.5f\n"
"#define INITIAL_M 0.5f\n"
"#define ADAPTATION_INTERVAL 10\n"
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
"__kernel void huddle_movement(\n"
"    __global float* population,\n"
"    __global float* fitness,\n"
"    __global const float* best_solution,\n"
"    __global const float* bounds,\n"
"    __global float* strategy_probs,\n"
"    __global int* strategy_success,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int iter,\n"
"    const int max_iter,\n"
"    uint seed,\n"
"    __local float* local_bounds,\n"
"    __local float* local_best)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id * dim + iter;\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        if (local_id < dim) {\n"
"            local_best[local_id] = best_solution[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        float T_prime = 2.0f - ((float)iter / max_iter);\n"
"        float R = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float r = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        int strategy = r < strategy_probs[0] ? STRATEGY_LINEAR : \n"
"                       (r < strategy_probs[0] + strategy_probs[1] ? STRATEGY_EXPONENTIAL : STRATEGY_CHAOTIC);\n"
"        float f, l, m_param;\n"
"        float t_norm = (float)iter / max_iter;\n"
"        if (strategy == STRATEGY_LINEAR) {\n"
"            f = 2.0f - t_norm * 1.5f;\n"
"            l = 1.5f - t_norm * 1.0f;\n"
"            m_param = 0.5f + t_norm * 0.3f;\n"
"        } else if (strategy == STRATEGY_EXPONENTIAL) {\n"
"            f = 2.0f * exp(-t_norm * 2.0f);\n"
"            l = 1.5f * exp(-t_norm * 3.0f);\n"
"            m_param = 0.5f * (1.0f + tanh(t_norm * 4.0f));\n"
"        } else {\n"
"            float x = 0.7f;\n"
"            for (int i = 0; i < (iter % 10); i++) {\n"
"                x = 4.0f * x * (1.0f - x);\n"
"            }\n"
"            f = 1.5f + x * 0.5f;\n"
"            l = 1.0f + x * 0.5f;\n"
"            m_param = 0.3f + x * 0.4f;\n"
"        }\n"
"        float S = m_param * exp(-t_norm / l) - exp(-t_norm);\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float D = fabs(f * lcg_rand_float(&local_seed, 0.0f, 1.0f) * local_best[d] - population[id * dim + d]);\n"
"            float new_pos = population[id * dim + d] + S * D * lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"            new_pos = max(local_bounds[2 * d], min(local_bounds[2 * d + 1], new_pos));\n"
"            population[id * dim + d] = new_pos;\n"
"        }\n"
"        atomic_add(&strategy_success[strategy], 1);\n"
"    }\n"
"}\n"
"__kernel void update_strategy_probabilities(\n"
"    __global float* strategy_probs,\n"
"    __global int* strategy_success)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id == 0) {\n"
"        float total_success = (float)(strategy_success[0] + strategy_success[1] + strategy_success[2]) + 1e-10f;\n"
"        for (int i = 0; i < STRATEGY_COUNT; i++) {\n"
"            strategy_probs[i] = (float)strategy_success[i] / total_success;\n"
"            strategy_probs[i] = max(0.1f, min(0.9f, strategy_probs[i]));\n"
"        }\n"
"        float sum_probs = strategy_probs[0] + strategy_probs[1] + strategy_probs[2];\n"
"        for (int i = 0; i < STRATEGY_COUNT; i++) {\n"
"            strategy_probs[i] /= sum_probs;\n"
"            strategy_success[i] = 0;\n"
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
"        if (local_id < offset && local_fitness[local_id + offset] < local_fitness[local_id]) {\n"
"            local_fitness[local_id] = local_fitness[local_id + offset];\n"
"            local_indices[local_id] = local_indices[local_id + offset];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (local_id == 0) {\n"
"        best_idx[get_group_id(0)] = local_indices[0];\n"
"    }\n"
"}\n";
void EPO_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, movement_kernel = NULL, strategy_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, strategy_probs_buffer = NULL, strategy_success_buffer = NULL, best_solution_buffer = NULL, best_idx_buffer = NULL;
    float* bounds_float = NULL, *population = NULL, *fitness = NULL, *strategy_probs = NULL;
    int* strategy_success = NULL;
    float* best_solution = NULL;
    int* best_idx_array = NULL;
    cl_event events[4] = {0};
    double start_time, end_time;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
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
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), or max_iter (%d)\n", dim, population_size, max_iter);
        exit(EXIT_FAILURE);
    }
    if (dim > 32) {
        fprintf(stderr, "Error: dim (%d) exceeds maximum supported value of 32\n", dim);
        exit(EXIT_FAILURE);
    }
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
        char platform_name[128] = "Unknown";
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        printf("Checking platform %u: %s\n", i, platform_name);
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            printf("No GPU devices found for platform %u\n", i);
            continue;
        }
        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        if (!devices) {
            fprintf(stderr, "Error: Memory allocation failed for devices\n");
            free(platforms);
            exit(EXIT_FAILURE);
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
                platform = platforms[i];
                device = devices[j];
                printf("Selected device: %s\n", device_name);
            }
        }
        free(devices);
        if (platform && device) break;
    }
    free(platforms);
    if (!platform || !device) {
        fprintf(stderr, "Error: No GPU device (Intel(R) Graphics [0x46a8]) found\n");
        exit(EXIT_FAILURE);
    }
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
    char platform_name[128] = "Unknown", device_name[128] = "Unknown";
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Selected platform: %s\n", platform_name);
    printf("Selected device: %s\n", device_name);
    size_t max_work_group_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying CL_DEVICE_MAX_WORK_GROUP_SIZE: %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = max_work_group_size < 64 ? max_work_group_size : 64;
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = 0.0;
    }
    program = clCreateProgramWithSource(context, 1, &epo_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating EPO program: %d\n", err);
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
            fprintf(stderr, "Error building EPO program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    movement_kernel = clCreateKernel(program, "huddle_movement", &err);
    strategy_kernel = clCreateKernel(program, "update_strategy_probabilities", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !movement_kernel || !strategy_kernel || !best_kernel) {
        fprintf(stderr, "Error creating EPO kernels: %d\n", err);
        goto cleanup;
    }
    size_t init_preferred_multiple, move_preferred_multiple, strat_preferred_multiple, best_preferred_multiple;
    err = clGetKernelWorkGroupInfo(init_kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &init_preferred_multiple, NULL);
    err |= clGetKernelWorkGroupInfo(movement_kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &move_preferred_multiple, NULL);
    err |= clGetKernelWorkGroupInfo(strategy_kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &strat_preferred_multiple, NULL);
    err |= clGetKernelWorkGroupInfo(best_kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &best_preferred_multiple, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Warning: Error querying kernel preferred work-group size: %d, using default %zu\n", err, local_work_size);
    } else {
        size_t min_multiple = init_preferred_multiple;
        min_multiple = min_multiple < move_preferred_multiple ? min_multiple : move_preferred_multiple;
        min_multiple = min_multiple < strat_preferred_multiple ? min_multiple : strat_preferred_multiple;
        min_multiple = min_multiple < best_preferred_multiple ? min_multiple : best_preferred_multiple;
        local_work_size = min_multiple < local_work_size ? min_multiple : local_work_size;
        printf("Preferred work-group size multiple: init=%zu, move=%zu, strat=%zu, best=%zu\n", 
               init_preferred_multiple, move_preferred_multiple, strat_preferred_multiple, best_preferred_multiple);
    }
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    population = (float*)malloc(population_size * dim * sizeof(float));
    fitness = (float*)malloc(population_size * sizeof(float));
    strategy_probs = (float*)malloc(STRATEGY_COUNT * sizeof(float));
    strategy_success = (int*)malloc(STRATEGY_COUNT * sizeof(int));
    best_solution = (float*)malloc(dim * sizeof(float));
    best_idx_array = (int*)malloc(population_size * sizeof(int));
    if (!bounds_float || !population || !fitness || !strategy_probs || !strategy_success || !best_solution || !best_idx_array) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < STRATEGY_COUNT; i++) {
        strategy_probs[i] = 0.3333333333333333f;
        strategy_success[i] = 0;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }
    for (int i = 0; i < dim; i++) {
        best_solution[i] = 0.0f;
    }
    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    strategy_probs_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, STRATEGY_COUNT * sizeof(float), NULL, &err);
    strategy_success_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, STRATEGY_COUNT * sizeof(int), NULL, &err);
    best_solution_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    opt->population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    opt->fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !strategy_probs_buffer || !strategy_success_buffer || 
        !best_solution_buffer || !opt->population_buffer || !opt->fitness_buffer || !best_idx_buffer) {
        fprintf(stderr, "Error creating EPO buffers: %d\n", err);
        goto cleanup;
    }
    err = clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, strategy_probs_buffer, CL_TRUE, 0, STRATEGY_COUNT * sizeof(float), strategy_probs, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, strategy_success_buffer, CL_TRUE, 0, STRATEGY_COUNT * sizeof(int), strategy_success, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, best_solution_buffer, CL_TRUE, 0, dim * sizeof(float), best_solution, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial buffers: %d\n", err);
        goto cleanup;
    }
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
    err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading initial population buffer: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < population_size; i++) {
        double* position = (double*)malloc(dim * sizeof(double));
        if (!position) {
            fprintf(stderr, "Error: Memory allocation failed for position\n");
            goto cleanup;
        }
        for (int d = 0; d < dim; d++) {
            position[d] = (double)population[i * dim + d];
        }
        fitness[i] = (float)objective_function(position);
        free(position);
    }
    err = clEnqueueWriteBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, &events[1]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial fitness buffer: %d\n", err);
        goto cleanup;
    }
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();
        err = clSetKernelArg(movement_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(movement_kernel, 1, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(movement_kernel, 2, sizeof(cl_mem), &best_solution_buffer);
        err |= clSetKernelArg(movement_kernel, 3, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(movement_kernel, 4, sizeof(cl_mem), &strategy_probs_buffer);
        err |= clSetKernelArg(movement_kernel, 5, sizeof(cl_mem), &strategy_success_buffer);
        err |= clSetKernelArg(movement_kernel, 6, sizeof(int), &dim);
        err |= clSetKernelArg(movement_kernel, 7, sizeof(int), &population_size);
        err |= clSetKernelArg(movement_kernel, 8, sizeof(int), &iter);
        err |= clSetKernelArg(movement_kernel, 9, sizeof(int), &max_iter);
        err |= clSetKernelArg(movement_kernel, 10, sizeof(uint), &seed);
        err |= clSetKernelArg(movement_kernel, 11, 2 * dim * sizeof(float), NULL);
        err |= clSetKernelArg(movement_kernel, 12, dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting movement kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, movement_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing movement kernel: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading population buffer: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < population_size; i++) {
            double* position = (double*)malloc(dim * sizeof(double));
            if (!position) {
                fprintf(stderr, "Error: Memory allocation failed for position\n");
                goto cleanup;
            }
            for (int d = 0; d < dim; d++) {
                position[d] = (double)population[i * dim + d];
            }
            fitness[i] = (float)objective_function(position);
            free(position);
        }
        err = clEnqueueWriteBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, &events[3]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            goto cleanup;
        }
        if (iter % ADAPTATION_INTERVAL == 0 && iter > 0) {
            err = clSetKernelArg(strategy_kernel, 0, sizeof(cl_mem), &strategy_probs_buffer);
            err |= clSetKernelArg(strategy_kernel, 1, sizeof(cl_mem), &strategy_success_buffer);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error setting strategy kernel args: %d\n", err);
                goto cleanup;
            }
            size_t strategy_global_work_size = local_work_size;
            err = clEnqueueNDRangeKernel(queue, strategy_kernel, 1, NULL, &strategy_global_work_size, &local_work_size, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error enqueuing strategy kernel: %d\n", err);
                goto cleanup;
            }
        }
        err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &best_idx_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(int), &population_size);
        err |= clSetKernelArg(best_kernel, 3, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 4, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing best kernel: %d\n", err);
            goto cleanup;
        }
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
            err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, best_idx * dim * sizeof(float), 
                                      dim * sizeof(float), best_solution, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best solution: %d\n", err);
                goto cleanup;
            }
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)best_solution[d];
            }
            err = clEnqueueWriteBuffer(queue, best_solution_buffer, CL_TRUE, 0, dim * sizeof(float), best_solution, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing best_solution buffer: %d\n", err);
                goto cleanup;
            }
        }
        cl_ulong time_start, time_end;
        double init_time = 0, fit_init_time = 0, move_time = 0, fit_time = 0;
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
                    if (err == CL_SUCCESS) fit_init_time = (time_end - time_start) / 1e6;
                }
            }
            if (events[2]) {
                err = clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                    if (err == CL_SUCCESS) move_time = (time_end - time_start) / 1e6;
                }
            }
            if (events[3]) {
                err = clGetEventProfilingInfo(events[3], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(events[3], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                    if (err == CL_SUCCESS) fit_time = (time_end - time_start) / 1e6;
                }
            }
            end_time = get_time_ms();
            printf("EPO|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | FitInit: %.3f ms | Move: %.3f ms | Fit: %.3f ms\n", 
                   iter + 1, opt->best_solution.fitness, end_time - start_time, 
                   init_time, fit_init_time, move_time, fit_time);
        } else {
            end_time = get_time_ms();
            printf("EPO|%5d -----> %9.16f | Total: %.3f ms | Profiling disabled\n", 
                   iter + 1, opt->best_solution.fitness, end_time - start_time);
        }
    }
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
        double* position = (double*)malloc(dim * sizeof(double));
        if (!position) {
            fprintf(stderr, "Error: Memory allocation failed for position\n");
            goto cleanup;
        }
        for (int d = 0; d < dim; d++) {
            position[d] = (double)population[i * dim + d];
            opt->population[i].position[d] = position[d];
        }
        opt->population[i].fitness = objective_function(position);
        free(position);
    }
cleanup:
    if (population) free(population);
    if (fitness) free(fitness);
    if (strategy_probs) free(strategy_probs);
    if (strategy_success) free(strategy_success);
    if (best_solution) free(best_solution);
    if (best_idx_array) free(best_idx_array);
    if (bounds_float) free(bounds_float);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (strategy_probs_buffer) clReleaseMemObject(strategy_probs_buffer);
    if (strategy_success_buffer) clReleaseMemObject(strategy_success_buffer);
    if (best_solution_buffer) clReleaseMemObject(best_solution_buffer);
    if (opt->population_buffer) clReleaseMemObject(opt->population_buffer);
    if (opt->fitness_buffer) clReleaseMemObject(opt->fitness_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (movement_kernel) clReleaseKernel(movement_kernel);
    if (strategy_kernel) clReleaseKernel(strategy_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 4; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (queue) clFinish(queue);
}
