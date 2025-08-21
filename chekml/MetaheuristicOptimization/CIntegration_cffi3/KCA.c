/* KCA.c - GPU-Optimized Key-based Clonal Algorithm with Flexible Objective Function */
#include "KCA.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Precompute bit scales for binary-to-continuous conversion
float BIT_SCALES[32] = {0};

// Initialize BIT_SCALES at program start
__attribute__((constructor)) void init_bit_scales() {
    for (int b = 0; b < 32; b++) {
        BIT_SCALES[b] = 1.0f / (1 << (b + 1));
    }
}

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
static const char* kca_kernel_source =
"#define KEY_INIT_PROB 0.5f\n"
"#define HALF_POPULATION_FACTOR 0.5f\n"
"#define SHUBERT_MIN -5.12f\n"
"#define SHUBERT_MAX 5.12f\n"
"#define FITNESS_SCALE 1.0f\n"
// Simple LCG random number generator
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
// Initialize binary keys
"__kernel void initialize_keys(\n"
"    __global int* keys,\n"
"    const int key_length,\n"
"    const int population_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size * key_length) {\n"
"        uint local_seed = seed + id;\n"
"        keys[id] = lcg_rand_float(&local_seed, 0.0f, 1.0f) < KEY_INIT_PROB ? 1 : 0;\n"
"    }\n"
"}\n"
// Binary to continuous conversion
"__kernel void binary_to_continuous(\n"
"    __global const int* keys,\n"
"    __global float* continuous_keys,\n"
"    __global const float* bit_scales,\n"
"    __global const float* bounds,\n"
"    const int key_length,\n"
"    const int dim,\n"
"    const int population_size)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        int bits_per_dim = key_length / dim;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float value = 0.0f;\n"
"            int key_offset = id * key_length + d * bits_per_dim;\n"
"            for (int b = 0; b < bits_per_dim; b++) {\n"
"                value += keys[key_offset + b] * bit_scales[b];\n"
"            }\n"
"            float pos = SHUBERT_MIN + value * (SHUBERT_MAX - SHUBERT_MIN);\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            continuous_keys[id * dim + d] = pos < min ? min : (pos > max ? max : pos);\n"
"        }\n"
"    }\n"
"}\n"
// Calculate probability factors
"__kernel void calculate_probability_factor(\n"
"    __global const int* keys,\n"
"    __global float* prob_factors,\n"
"    const int key_length,\n"
"    const int population_size,\n"
"    const int half_pop)\n"
"{\n"
"    int j = get_global_id(0);\n"
"    if (j < key_length) {\n"
"        float tooth_sum = 0.0f;\n"
"        for (int i = 0; i < half_pop; i++) {\n"
"            tooth_sum += keys[i * key_length + j];\n"
"        }\n"
"        float average = tooth_sum / half_pop;\n"
"        float prob = 1.0f - average;\n"
"        for (int i = 0; i < half_pop; i++) {\n"
"            prob_factors[i * key_length + j] = prob;\n"
"        }\n"
"    }\n"
"}\n"
// Generate new keys
"__kernel void generate_new_keys(\n"
"    __global int* keys,\n"
"    __global const float* prob_factors,\n"
"    const int key_length,\n"
"    const int population_size,\n"
"    const int half_pop,\n"
"    const int use_kca1,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < half_pop) {\n"
"        int new_key_idx = (half_pop + id) * key_length;\n"
"        int old_key_idx = id * key_length;\n"
"        uint local_seed = seed + id * key_length;\n"
"        for (int j = 0; j < key_length; j++) {\n"
"            keys[new_key_idx + j] = keys[old_key_idx + j];\n"
"            float random_num = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"            float prob = prob_factors[id * key_length + j];\n"
"            if ((use_kca1 && random_num < prob) || (!use_kca1 && random_num > prob)) {\n"
"                keys[new_key_idx + j] = 1 - keys[new_key_idx + j];\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
// Sort population by fitness
"__kernel void find_best_and_sort(\n"
"    __global const float* fitness,\n"
"    __global int* sorted_indices,\n"
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
"    if (id < population_size) {\n"
"        sorted_indices[id] = local_indices[local_id];\n"
"    }\n"
"}\n"
// Reorganize keys and positions
"__kernel void reorganize_keys_and_positions(\n"
"    __global const int* keys,\n"
"    __global int* temp_keys,\n"
"    __global const float* continuous_keys,\n"
"    __global float* population_positions,\n"
"    __global const float* fitness,\n"
"    __global float* population_fitness,\n"
"    __global const int* sorted_indices,\n"
"    __global const float* bounds,\n"
"    const int key_length,\n"
"    const int dim,\n"
"    const int population_size)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        int src_idx = sorted_indices[id];\n"
"        for (int j = 0; j < key_length; j++) {\n"
"            temp_keys[id * key_length + j] = keys[src_idx * key_length + j];\n"
"        }\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float pos = continuous_keys[src_idx * dim + d];\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            population_positions[id * dim + d] = pos < min ? min : (pos > max ? max : pos);\n"
"        }\n"
"        population_fitness[id] = fitness[src_idx];\n"
"    }\n"
"}\n"
// Enforce bound constraints
"__kernel void enforce_bounds(\n"
"    __global float* population_positions,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            float pos = population_positions[id * dim + d];\n"
"            population_positions[id * dim + d] = fmax(min, fmin(max, pos));\n"
"        }\n"
"    }\n"
"}\n";

void KCA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, binary_to_cont_kernel = NULL, prob_factor_kernel = NULL;
    cl_kernel generate_keys_kernel = NULL, sort_kernel = NULL, reorganize_kernel = NULL, bounds_kernel = NULL;
    cl_mem keys_buffer = NULL, continuous_keys_buffer = NULL, prob_factors_buffer = NULL;
    cl_mem sorted_indices_buffer = NULL, temp_keys_buffer = NULL, best_idx_buffer = NULL;
    cl_mem bit_scales_buffer = NULL, bounds_buffer = NULL;
    float* continuous_keys = NULL, *fitness = NULL, *bit_scales_float = NULL, *bounds_float = NULL;
    int* keys = NULL, *sorted_indices = NULL, *best_idx_array = NULL;
    cl_event events[7] = {0};
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
    const int key_length = dim * BITS_PER_DIM;
    const int half_pop = (int)(population_size * HALF_POPULATION_FACTOR);
    const int use_kca1 = 1;

    if (dim < 1 || population_size < KCA_MIN_POP_SIZE || population_size > KCA_MAX_POP_SIZE || 
        max_iter < 1 || key_length < KCA_MIN_KEY_LENGTH || key_length > KCA_MAX_KEY_LENGTH) {
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), max_iter (%d), or key_length (%d)\n", 
                dim, population_size, max_iter, key_length);
        exit(EXIT_FAILURE);
    }
    if (dim > 32) {
        fprintf(stderr, "Error: dim (%d) exceeds maximum supported value of 32\n", dim);
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
        char platform_name[128] = "Unknown";
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        printf("Checking platform %u: %s\n", i, platform_name);
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            continue;
        }
        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        if (!devices) {
            continue;
        }
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
        if (err != CL_SUCCESS) {
            free(devices);
            continue;
        }
        for (cl_uint j = 0; j < num_devices; j++) {
            char device_name[128] = "Unknown";
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            printf("  Device %u: %s\n", j, device_name);
            platform = platforms[i];
            device = devices[j];
            printf("Selected device: %s\n", device_name);
            break;
        }
        free(devices);
        if (platform && device) break;
    }
    free(platforms);
    if (!platform || !device) {
        fprintf(stderr, "Error: No suitable GPU device found\n");
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
    program = clCreateProgramWithSource(context, 1, &kca_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating KCA program: %d\n", err);
        goto cleanup;
    }
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building KCA program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        goto cleanup;
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_keys", &err);
    binary_to_cont_kernel = clCreateKernel(program, "binary_to_continuous", &err);
    prob_factor_kernel = clCreateKernel(program, "calculate_probability_factor", &err);
    generate_keys_kernel = clCreateKernel(program, "generate_new_keys", &err);
    sort_kernel = clCreateKernel(program, "find_best_and_sort", &err);
    reorganize_kernel = clCreateKernel(program, "reorganize_keys_and_positions", &err);
    bounds_kernel = clCreateKernel(program, "enforce_bounds", &err);
    if (err != CL_SUCCESS || !init_kernel || !binary_to_cont_kernel || !prob_factor_kernel ||
        !generate_keys_kernel || !sort_kernel || !reorganize_kernel || !bounds_kernel) {
        fprintf(stderr, "Error creating KCA kernels: %d\n", err);
        goto cleanup;
    }

    // Allocate host memory
    keys = (int*)malloc(population_size * key_length * sizeof(int));
    continuous_keys = (float*)malloc(population_size * dim * sizeof(float));
    fitness = (float*)malloc(population_size * sizeof(float));
    sorted_indices = (int*)malloc(population_size * sizeof(int));
    best_idx_array = (int*)malloc(population_size * sizeof(int));
    bit_scales_float = (float*)malloc(32 * sizeof(float));
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    temp_position = (double*)malloc(dim * sizeof(double));
    if (!keys || !continuous_keys || !fitness || !sorted_indices || !best_idx_array ||
        !bit_scales_float || !bounds_float || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 32; i++) {
        bit_scales_float[i] = BIT_SCALES[i];
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }

    // Create buffers
    keys_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * key_length * sizeof(int), NULL, &err);
    continuous_keys_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    prob_factors_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, half_pop * key_length * sizeof(float), NULL, &err);
    sorted_indices_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    temp_keys_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * key_length * sizeof(int), NULL, &err);
    best_idx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    bit_scales_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 32 * sizeof(float), NULL, &err);
    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    opt->population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    opt->fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !keys_buffer || !continuous_keys_buffer || !prob_factors_buffer ||
        !sorted_indices_buffer || !temp_keys_buffer || !best_idx_buffer || !bit_scales_buffer ||
        !bounds_buffer || !opt->population_buffer || !opt->fitness_buffer) {
        fprintf(stderr, "Error creating KCA buffers: %d\n", err);
        goto cleanup;
    }

    // Write bit scales and bounds
    err = clEnqueueWriteBuffer(queue, bit_scales_buffer, CL_TRUE, 0, 32 * sizeof(float), bit_scales_float, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bit_scales or bounds buffer: %d\n", err);
        goto cleanup;
    }

    // Initialize keys
    uint seed = (uint)time(NULL);
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &keys_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(int), &key_length);
    err |= clSetKernelArg(init_kernel, 2, sizeof(int), &population_size);
    err |= clSetKernelArg(init_kernel, 3, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting init kernel args: %d\n", err);
        goto cleanup;
    }
    size_t init_global_work_size = ((population_size * key_length + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, &init_global_work_size, &local_work_size, 0, NULL, &events[0]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel: %d\n", err);
        goto cleanup;
    }

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Convert binary keys to continuous
        err = clSetKernelArg(binary_to_cont_kernel, 0, sizeof(cl_mem), &keys_buffer);
        err |= clSetKernelArg(binary_to_cont_kernel, 1, sizeof(cl_mem), &continuous_keys_buffer);
        err |= clSetKernelArg(binary_to_cont_kernel, 2, sizeof(cl_mem), &bit_scales_buffer);
        err |= clSetKernelArg(binary_to_cont_kernel, 3, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(binary_to_cont_kernel, 4, sizeof(int), &key_length);
        err |= clSetKernelArg(binary_to_cont_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(binary_to_cont_kernel, 6, sizeof(int), &population_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting binary_to_continuous kernel args: %d\n", err);
            goto cleanup;
        }
        size_t binary_to_cont_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, binary_to_cont_kernel, 1, NULL, &binary_to_cont_global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing binary_to_continuous kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate fitness on host (assuming objective_function is CPU-based)
        err = clEnqueueReadBuffer(queue, continuous_keys_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), continuous_keys, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading continuous_keys buffer: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < population_size; i++) {
            for (int d = 0; d < dim; d++) {
                temp_position[d] = (double)continuous_keys[i * dim + d];
            }
            fitness[i] = (float)(objective_function(temp_position) * FITNESS_SCALE);
        }
        err = clEnqueueWriteBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            goto cleanup;
        }

        // Find best and sort
        err = clSetKernelArg(sort_kernel, 0, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(sort_kernel, 1, sizeof(cl_mem), &sorted_indices_buffer);
        err |= clSetKernelArg(sort_kernel, 2, sizeof(cl_mem), &best_idx_buffer);
        err |= clSetKernelArg(sort_kernel, 3, sizeof(int), &population_size);
        err |= clSetKernelArg(sort_kernel, 4, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(sort_kernel, 5, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting sort kernel args: %d\n", err);
            goto cleanup;
        }
        size_t sort_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, sort_kernel, 1, NULL, &sort_global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing sort kernel: %d\n", err);
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
            err = clEnqueueReadBuffer(queue, continuous_keys_buffer, CL_TRUE, best_idx * dim * sizeof(float), 
                                      dim * sizeof(float), best_solution, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best solution: %d\n", err);
                free(best_solution);
                goto cleanup;
            }
            for (int d = 0; d < dim; d++) {
                float pos = best_solution[d];
                float min = bounds_float[2 * d];
                float max = bounds_float[2 * d + 1];
                opt->best_solution.position[d] = (double)(pos < min ? min : (pos > max ? max : pos));
            }
            free(best_solution);
        }

        // Reorganize keys and positions
        err = clSetKernelArg(reorganize_kernel, 0, sizeof(cl_mem), &keys_buffer);
        err |= clSetKernelArg(reorganize_kernel, 1, sizeof(cl_mem), &temp_keys_buffer);
        err |= clSetKernelArg(reorganize_kernel, 2, sizeof(cl_mem), &continuous_keys_buffer);
        err |= clSetKernelArg(reorganize_kernel, 3, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(reorganize_kernel, 4, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(reorganize_kernel, 5, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(reorganize_kernel, 6, sizeof(cl_mem), &sorted_indices_buffer);
        err |= clSetKernelArg(reorganize_kernel, 7, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(reorganize_kernel, 8, sizeof(int), &key_length);
        err |= clSetKernelArg(reorganize_kernel, 9, sizeof(int), &dim);
        err |= clSetKernelArg(reorganize_kernel, 10, sizeof(int), &population_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting reorganize kernel args: %d\n", err);
            goto cleanup;
        }
        size_t reorganize_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, reorganize_kernel, 1, NULL, &reorganize_global_work_size, &local_work_size, 0, NULL, &events[3]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing reorganize kernel: %d\n", err);
            goto cleanup;
        }

        // Copy temp_keys back to keys
        err = clEnqueueCopyBuffer(queue, temp_keys_buffer, keys_buffer, 0, 0, population_size * key_length * sizeof(int), 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error copying temp_keys to keys: %d\n", err);
            goto cleanup;
        }

        // Calculate probability factors
        err = clSetKernelArg(prob_factor_kernel, 0, sizeof(cl_mem), &keys_buffer);
        err |= clSetKernelArg(prob_factor_kernel, 1, sizeof(cl_mem), &prob_factors_buffer);
        err |= clSetKernelArg(prob_factor_kernel, 2, sizeof(int), &key_length);
        err |= clSetKernelArg(prob_factor_kernel, 3, sizeof(int), &population_size);
        err |= clSetKernelArg(prob_factor_kernel, 4, sizeof(int), &half_pop);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting prob_factor kernel args: %d\n", err);
            goto cleanup;
        }
        size_t prob_factor_global_work_size = ((key_length + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, prob_factor_kernel, 1, NULL, &prob_factor_global_work_size, &local_work_size, 0, NULL, &events[4]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing prob_factor kernel: %d\n", err);
            goto cleanup;
        }

        // Generate new keys
        err = clSetKernelArg(generate_keys_kernel, 0, sizeof(cl_mem), &keys_buffer);
        err |= clSetKernelArg(generate_keys_kernel, 1, sizeof(cl_mem), &prob_factors_buffer);
        err |= clSetKernelArg(generate_keys_kernel, 2, sizeof(int), &key_length);
        err |= clSetKernelArg(generate_keys_kernel, 3, sizeof(int), &population_size);
        err |= clSetKernelArg(generate_keys_kernel, 4, sizeof(int), &half_pop);
        err |= clSetKernelArg(generate_keys_kernel, 5, sizeof(int), &use_kca1);
        err |= clSetKernelArg(generate_keys_kernel, 6, sizeof(uint), &seed);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting generate_keys kernel args: %d\n", err);
            goto cleanup;
        }
        size_t generate_keys_global_work_size = ((half_pop + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, generate_keys_kernel, 1, NULL, &generate_keys_global_work_size, &local_work_size, 0, NULL, &events[5]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing generate_keys kernel: %d\n", err);
            goto cleanup;
        }

        // Enforce bounds
        err = clSetKernelArg(bounds_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(bounds_kernel, 1, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(bounds_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(bounds_kernel, 3, sizeof(int), &population_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting bounds kernel args: %d\n", err);
            goto cleanup;
        }
        size_t bounds_global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, bounds_kernel, 1, NULL, &bounds_global_work_size, &local_work_size, 0, NULL, &events[6]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing bounds kernel: %d\n", err);
            goto cleanup;
        }

        // Profiling
        cl_ulong time_start, time_end;
        double init_time = 0, binary_time = 0, sort_time = 0, reorganize_time = 0, prob_time = 0, generate_time = 0, bounds_time = 0;
        cl_ulong queue_properties;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(cl_ulong), &queue_properties, NULL);
        if (err == CL_SUCCESS && (queue_properties & CL_QUEUE_PROFILING_ENABLE)) {
            for (int i = 0; i < 7; i++) {
                if (events[i]) {
                    err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                    if (err == CL_SUCCESS) {
                        err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                        if (err == CL_SUCCESS) {
                            double time_ms = (time_end - time_start) / 1e6;
                            if (i == 0) init_time = time_ms;
                            else if (i == 1) binary_time = time_ms;
                            else if (i == 2) sort_time = time_ms;
                            else if (i == 3) reorganize_time = time_ms;
                            else if (i == 4) prob_time = time_ms;
                            else if (i == 5) generate_time = time_ms;
                            else if (i == 6) bounds_time = time_ms;
                        }
                    }
                }
            }
            end_time = get_time_ms();
            printf("KCA|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Binary: %.3f ms | Sort: %.3f ms | Reorganize: %.3f ms | Prob: %.3f ms | Generate: %.3f ms | Bounds: %.3f ms\n",
                   iter + 1, opt->best_solution.fitness, end_time - start_time,
                   init_time, binary_time, sort_time, reorganize_time, prob_time, generate_time, bounds_time);
        } else {
            end_time = get_time_ms();
            printf("KCA|%5d -----> %9.16f | Total: %.3f ms | Profiling disabled\n",
                   iter + 1, opt->best_solution.fitness, end_time - start_time);
        }
    }

    // Update CPU-side population
    err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), continuous_keys, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final population buffer: %d\n", err);
        goto cleanup;
    }
    err = clEnqueueReadBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final fitness buffer: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: Population[%d].position is null during final update\n", i);
            goto cleanup;
        }
        for (int d = 0; d < dim; d++) {
            float pos = continuous_keys[i * dim + d];
            float min = bounds_float[2 * d];
            float max = bounds_float[2 * d + 1];
            opt->population[i].position[d] = (double)(pos < min ? min : (pos > max ? max : pos));
        }
        opt->population[i].fitness = (double)fitness[i];
    }

cleanup:
    if (keys) free(keys);
    if (continuous_keys) free(continuous_keys);
    if (fitness) free(fitness);
    if (sorted_indices) free(sorted_indices);
    if (best_idx_array) free(best_idx_array);
    if (bit_scales_float) free(bit_scales_float);
    if (bounds_float) free(bounds_float);
    if (temp_position) free(temp_position);

    if (keys_buffer) clReleaseMemObject(keys_buffer);
    if (continuous_keys_buffer) clReleaseMemObject(continuous_keys_buffer);
    if (prob_factors_buffer) clReleaseMemObject(prob_factors_buffer);
    if (sorted_indices_buffer) clReleaseMemObject(sorted_indices_buffer);
    if (temp_keys_buffer) clReleaseMemObject(temp_keys_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (bit_scales_buffer) clReleaseMemObject(bit_scales_buffer);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (opt->population_buffer) clReleaseMemObject(opt->population_buffer);
    if (opt->fitness_buffer) clReleaseMemObject(opt->fitness_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (binary_to_cont_kernel) clReleaseKernel(binary_to_cont_kernel);
    if (prob_factor_kernel) clReleaseKernel(prob_factor_kernel);
    if (generate_keys_kernel) clReleaseKernel(generate_keys_kernel);
    if (sort_kernel) clReleaseKernel(sort_kernel);
    if (reorganize_kernel) clReleaseKernel(reorganize_kernel);
    if (bounds_kernel) clReleaseKernel(bounds_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 7; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (queue) clFinish(queue);
}
