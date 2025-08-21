/* FHO.c - GPU-Optimized Fire Hawk Optimization with Robust Memory Safety and Enhanced Exploration */
#include "FHO.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
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
static const char* fho_kernel_source =
"#define MIN_FIREHAWKS 1\n"
"#define MAX_FIREHAWKS_RATIO 0.1f\n"
"#define IR_MIN 0.0f\n"
"#define IR_MAX 1.0f\n"
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
"__kernel void initialize_population(\n"
"    __global float* population,\n"
"    __global float* fitness,\n"
"    __global uint* rng_states,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id >= population_size) return;\n"
"    uint rng_state = rng_states[id];\n"
"    if (local_id < dim * 2) {\n"
"        local_bounds[local_id] = bounds[local_id];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (id < population_size) {\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float min = local_bounds[2 * d];\n"
"            float max = local_bounds[2 * d + 1];\n"
"            population[id * dim + d] = lcg_rand_float(&rng_state, min, max);\n"
"        }\n"
"        fitness[id] = INFINITY;\n"
"    }\n"
"    rng_states[id] = rng_state;\n"
"}\n"
"__kernel void update_firehawks(\n"
"    __global float* population,\n"
"    __global const float* best_solution,\n"
"    __global const float* bounds,\n"
"    __global uint* rng_states,\n"
"    const int dim,\n"
"    const int num_firehawks)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id >= num_firehawks) return;\n"
"    uint rng_state = rng_states[id];\n"
"    int other_fh_idx = lcg_rand(&rng_state) % num_firehawks;\n"
"    float ir1 = lcg_rand_float(&rng_state, IR_MIN, IR_MAX);\n"
"    float ir2 = lcg_rand_float(&rng_state, IR_MIN, IR_MAX);\n"
"    float perturb = lcg_rand_float(&rng_state, -3.0f, 3.0f);\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float new_pos = population[id * dim + d] + \n"
"                       (ir1 * best_solution[d] - ir2 * population[other_fh_idx * dim + d]) + perturb;\n"
"        new_pos = fmax(bounds[2 * d], fmin(bounds[2 * d + 1], new_pos));\n"
"        population[id * dim + d] = new_pos;\n"
"    }\n"
"    rng_states[id] = rng_state;\n"
"}\n"
"__kernel void compute_safe_points(\n"
"    __global const float* population,\n"
"    __global const int* prey_counts,\n"
"    __global const int* prey_indices,\n"
"    __global float* local_safe_points,\n"
"    __global float* global_safe_point,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int num_firehawks,\n"
"    const int prey_size,\n"
"    __local float* local_sums)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    if (id < num_firehawks) {\n"
"        int count = prey_counts[id];\n"
"        for (int d = local_id; d < dim; d += local_size) {\n"
"            float sum = 0.0f;\n"
"            for (int j = 0; j < count; j++) {\n"
"                sum += population[prey_indices[id * prey_size + j] * dim + d];\n"
"            }\n"
"            local_safe_points[id * dim + d] = count > 0 ? sum / count : 0.0f;\n"
"        }\n"
"    }\n"
"    for (int d = local_id; d < dim; d += local_size) {\n"
"        float sum = 0.0f;\n"
"        for (int i = 0; i < population_size; i++) {\n"
"            sum += population[i * dim + d];\n"
"        }\n"
"        global_safe_point[d] = sum / population_size;\n"
"    }\n"
"}\n"
"__kernel void update_prey(\n"
"    __global float* population,\n"
"    __global const int* prey_counts,\n"
"    __global const int* prey_indices,\n"
"    __global const float* local_safe_points,\n"
"    __global const float* global_safe_point,\n"
"    __global const float* bounds,\n"
"    __global uint* rng_states,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int num_firehawks,\n"
"    const int prey_size,\n"
"    const int total_prey,\n"
"    const float select_threshold)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id >= total_prey) return;\n"
"    uint rng_state = rng_states[id];\n"
"    int fh_idx = 0, prey_offset = id;\n"
"    for (int i = 0; i < num_firehawks; i++) {\n"
"        if (prey_offset < prey_counts[i]) {\n"
"            fh_idx = i;\n"
"            break;\n"
"        }\n"
"        prey_offset -= prey_counts[i];\n"
"    }\n"
"    int prey_idx = prey_indices[fh_idx * prey_size + prey_offset];\n"
"    float ir1 = lcg_rand_float(&rng_state, IR_MIN, IR_MAX);\n"
"    float ir2 = lcg_rand_float(&rng_state, IR_MIN, IR_MAX);\n"
"    float select = lcg_rand_float(&rng_state, 0.0f, 1.0f);\n"
"    int rand_fh_idx = lcg_rand(&rng_state) % population_size;\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float pos1 = population[prey_idx * dim + d] + \n"
"                     (ir1 * population[fh_idx * dim + d] - ir2 * local_safe_points[fh_idx * dim + d]);\n"
"        pos1 = fmax(bounds[2 * d], fmin(bounds[2 * d + 1], pos1));\n"
"        float pos2 = population[prey_idx * dim + d] + \n"
"                     (ir1 * population[rand_fh_idx * dim + d] - ir2 * global_safe_point[d]);\n"
"        pos2 = fmax(bounds[2 * d], fmin(bounds[2 * d + 1], pos2));\n"
"        population[prey_idx * dim + d] = select < select_threshold ? pos1 : pos2;\n"
"    }\n"
"    rng_states[id] = rng_state;\n"
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

// Main Optimization Function
void FHO_optimize(Optimizer* opt, double (*objective_function)(double*)) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, fh_kernel = NULL, safe_points_kernel = NULL;
    cl_kernel prey_kernel = NULL, best_kernel = NULL;
    cl_command_queue queue = NULL;
    cl_mem population_buffer = NULL, fitness_buffer = NULL, bounds_buffer = NULL;
    cl_mem best_solution_buffer = NULL, prey_counts_buffer = NULL, prey_indices_buffer = NULL;
    cl_mem local_safe_points_buffer = NULL, global_safe_point_buffer = NULL, rng_states_buffer = NULL;
    float *population = NULL, *fitness = NULL, *bounds = NULL, *best_solution = NULL;
    int *prey_counts = NULL, *prey_indices = NULL;
    uint *rng_states = NULL;
    double *cpu_position = NULL;
    float *distances = NULL;
    int *sorted_indices = NULL;
    cl_event events[5] = {0};
    double start_time, end_time;
    double last_best_fitness = INFINITY;
    int stagnant_count = 0;

    // Validate input
    if (!opt || !opt->population || !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure, null pointers, or missing objective function\n");
        goto cleanup;
    }
    const int dim = opt->dim;
    const int population_size = opt->population_size;
    const int max_iter = opt->max_iter;
    if (dim < 1 || population_size < 1 || max_iter < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), or max_iter (%d)\n", 
                dim, population_size, max_iter);
        goto cleanup;
    }
    if (dim > 32) {
        fprintf(stderr, "Error: dim (%d) exceeds maximum supported value of 32\n", dim);
        goto cleanup;
    }
    for (int d = 0; d < dim; d++) {
        if (opt->bounds[d * 2] >= opt->bounds[d * 2 + 1]) {
            fprintf(stderr, "Error: Invalid bounds for dimension %d: [%f, %f]\n", 
                    d, opt->bounds[d * 2], opt->bounds[d * 2 + 1]);
            goto cleanup;
        }
    }

    // Check GPU memory limits
    cl_ulong max_alloc_size, global_mem_size;
    err = clGetDeviceInfo(opt->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc_size, NULL);
    err |= clGetDeviceInfo(opt->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying device memory info: %d\n", err);
        goto cleanup;
    }
    size_t max_firehawks = (size_t)(ceil(population_size * MAX_FIREHAWKS_RATIO)) + 1;
    size_t prey_size = population_size - max_firehawks;
    size_t prey_indices_size = max_firehawks * prey_size * sizeof(int);
    size_t total_buffer_size = population_size * dim * sizeof(float) + // population_buffer
                              population_size * sizeof(float) +       // fitness_buffer
                              2 * dim * sizeof(float) +               // bounds_buffer
                              dim * sizeof(float) +                   // best_solution_buffer
                              population_size * sizeof(int) +         // prey_counts_buffer
                              prey_indices_size +                     // prey_indices_buffer
                              population_size * dim * sizeof(float) + // local_safe_points_buffer
                              dim * sizeof(float) +                   // global_safe_point_buffer
                              population_size * sizeof(uint);         // rng_states_buffer
    if (total_buffer_size > global_mem_size || prey_indices_size > max_alloc_size) {
        fprintf(stderr, "Error: Required buffer size (%zu bytes) exceeds GPU memory limits (global: %zu, max alloc: %zu)\n",
                total_buffer_size, global_mem_size, max_alloc_size);
        goto cleanup;
    }
    fprintf(stderr, "Memory allocation: population=%zu bytes, fitness=%zu bytes, prey_indices=%zu bytes\n",
            population_size * dim * sizeof(float), population_size * sizeof(float), prey_indices_size);

    // Create command queue with profiling
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(opt->context, opt->device, props, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating command queue: %d\n", err);
        goto cleanup;
    }

    // Query device capabilities
    size_t max_work_group_size;
    err = clGetDeviceInfo(opt->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying CL_DEVICE_MAX_WORK_GROUP_SIZE: %d\n", err);
        goto cleanup;
    }
    size_t local_work_size = max_work_group_size < 32 ? max_work_group_size : 32;

    // Create OpenCL program
    program = clCreateProgramWithSource(opt->context, 1, &fho_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating FHO program: %d\n", err);
        goto cleanup;
    }
    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)calloc(log_size, 1);
        if (log) {
            fprintf(stderr, "Allocating %zu bytes for program build log\n", log_size);
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building FHO program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        goto cleanup;
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    fh_kernel = clCreateKernel(program, "update_firehawks", &err);
    safe_points_kernel = clCreateKernel(program, "compute_safe_points", &err);
    prey_kernel = clCreateKernel(program, "update_prey", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating FHO kernels: %d\n", err);
        goto cleanup;
    }

    // Allocate GPU buffers
    population_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    best_solution_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    prey_counts_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    prey_indices_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, max_firehawks * prey_size * sizeof(int), NULL, &err);
    local_safe_points_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    global_safe_point_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    rng_states_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(uint), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating FHO buffers: %d\n", err);
        goto cleanup;
    }

    // Allocate host memory with calloc for zero-initialization
    population = (float*)calloc(population_size * dim, sizeof(float));
    fitness = (float*)calloc(population_size, sizeof(float));
    bounds = (float*)calloc(2 * dim, sizeof(float));
    best_solution = (float*)calloc(dim, sizeof(float));
    prey_counts = (int*)calloc(population_size, sizeof(int));
    prey_indices = (int*)calloc(max_firehawks * prey_size, sizeof(int));
    rng_states = (uint*)calloc(population_size, sizeof(uint));
    cpu_position = (double*)calloc(dim, sizeof(double));
    distances = (float*)calloc(population_size, sizeof(float));
    sorted_indices = (int*)calloc(population_size, sizeof(int));
    if (!population || !fitness || !bounds || !best_solution || !prey_counts || !prey_indices || 
        !rng_states || !cpu_position || !distances || !sorted_indices) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    fprintf(stderr, "Allocated: population=%p, fitness=%p, bounds=%p, best_solution=%p, prey_counts=%p, prey_indices=%p, rng_states=%p, cpu_position=%p, distances=%p, sorted_indices=%p\n",
            (void*)population, (void*)fitness, (void*)bounds, (void*)best_solution, (void*)prey_counts, 
            (void*)prey_indices, (void*)rng_states, (void*)cpu_position, (void*)distances, (void*)sorted_indices);
    // Initialize prey_indices to -1
    for (size_t i = 0; i < max_firehawks * prey_size; i++) {
        prey_indices[i] = -1;
    }

    // Initialize host data
    srand((unsigned int)time(NULL));
    for (int d = 0; d < dim; d++) {
        bounds[d * 2] = (float)opt->bounds[d * 2];
        bounds[d * 2 + 1] = (float)opt->bounds[d * 2 + 1];
    }
    for (int i = 0; i < population_size; i++) {
        rng_states[i] = (uint)(rand() ^ (i + 1));
        fitness[i] = INFINITY;
        for (int d = 0; d < dim; d++) {
            if (i * dim + d >= population_size * dim) {
                fprintf(stderr, "Error: population index %d out of bounds, max=%d\n", 
                        i * dim + d, population_size * dim);
                goto cleanup;
            }
            population[i * dim + d] = (float)opt->population[i].position[d];
        }
    }
    for (int d = 0; d < dim; d++) {
        if (d >= dim) {
            fprintf(stderr, "Error: best_solution index %d out of bounds, max=%d\n", d, dim);
            goto cleanup;
        }
        best_solution[d] = (float)opt->best_solution.position[d];
    }

    // Write initial data to GPU
    err = clEnqueueWriteBuffer(queue, population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, best_solution_buffer, CL_TRUE, 0, dim * sizeof(float), best_solution, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, prey_counts_buffer, CL_TRUE, 0, population_size * sizeof(int), prey_counts, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, rng_states_buffer, CL_TRUE, 0, population_size * sizeof(uint), rng_states, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial buffers: %d\n", err);
        goto cleanup;
    }

    // Initialize population
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &fitness_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(cl_mem), &rng_states_buffer);
    err |= clSetKernelArg(init_kernel, 3, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 4, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 5, sizeof(int), &population_size);
    err |= clSetKernelArg(init_kernel, 6, 2 * dim * sizeof(float), NULL);
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
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error finishing queue after init kernel: %d\n", err);
        goto cleanup;
    }

    // Evaluate initial fitness on CPU
    err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading initial population: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            if (i * dim + d >= population_size * dim) {
                fprintf(stderr, "Error: population index %d out of bounds, max=%d\n", 
                        i * dim + d, population_size * dim);
                goto cleanup;
            }
            cpu_position[d] = (double)population[i * dim + d];
        }
        fitness[i] = (float)objective_function(cpu_position);
        if (!isfinite(fitness[i])) {
            fprintf(stderr, "Warning: Non-finite fitness for individual %d: %f\n", i, fitness[i]);
            fitness[i] = INFINITY;
        }
    }
    err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial fitness: %d\n", err);
        goto cleanup;
    }

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < population_size; i++) {
        if (i >= population_size) {
            fprintf(stderr, "Error: fitness index %d out of bounds, max=%d\n", i, population_size);
            goto cleanup;
        }
        if (fitness[i] < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)fitness[i];
            for (int d = 0; d < dim; d++) {
                if (i * dim + d >= population_size * dim) {
                    fprintf(stderr, "Error: population index %d out of bounds, max=%d\n", 
                            i * dim + d, population_size * dim);
                    goto cleanup;
                }
                opt->best_solution.position[d] = (double)population[i * dim + d];
                best_solution[d] = population[i * dim + d];
            }
        }
    }
    err = clEnqueueWriteBuffer(queue, best_solution_buffer, CL_TRUE, 0, dim * sizeof(float), best_solution, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial best solution: %d\n", err);
        goto cleanup;
    }

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Determine number of firehawks
        size_t num_firehawks = MIN_FIREHAWKS + (rand() % (max_firehawks - MIN_FIREHAWKS + 1));
        if (num_firehawks > max_firehawks || num_firehawks > population_size || num_firehawks < MIN_FIREHAWKS) {
            fprintf(stderr, "Error: Invalid num_firehawks %zu, max=%zu, population_size=%d, iter %d\n", 
                    num_firehawks, max_firehawks, population_size, iter);
            num_firehawks = max_firehawks < population_size ? max_firehawks : population_size;
        }
        fprintf(stderr, "Iter %d: max_firehawks=%zu, num_firehawks=%zu\n", iter, max_firehawks, num_firehawks);

        // Compute dynamic select threshold
        float select_threshold = 0.7f - 0.5f * ((float)iter / max_iter);

        // Assign prey on CPU
        err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading population for prey assignment: %d\n", err);
            goto cleanup;
        }
        memset(prey_counts, 0, population_size * sizeof(int));
        prey_size = population_size - num_firehawks;
        if (prey_size <= 0 || prey_size > population_size) {
            fprintf(stderr, "Error: Invalid prey_size %zu, num_firehawks=%zu, population_size=%d, iter %d\n",
                    prey_size, num_firehawks, population_size, iter);
            goto cleanup;
        }
        size_t total_prey = 0;
        fprintf(stderr, "Iter %d: num_firehawks=%zu, prey_size=%zu, select_threshold=%f\n", 
                iter, num_firehawks, prey_size, select_threshold);
        for (size_t i = 0; i < num_firehawks; i++) {
            if (i >= max_firehawks || i >= population_size) {
                fprintf(stderr, "Error: firehawk index %zu out of bounds, max=%zu, population_size=%d, iter %d\n", 
                        i, max_firehawks, population_size, iter);
                goto cleanup;
            }
            // Initialize distances and indices
            for (size_t j = 0; j < prey_size; j++) {
                if (j >= population_size) {
                    fprintf(stderr, "Error: distances index %zu out of bounds, max=%d, iter %d\n", 
                            j, population_size, iter);
                    goto cleanup;
                }
                if (i >= population_size || num_firehawks + j >= population_size) {
                    fprintf(stderr, "Error: population index %zu or %zu out of bounds, max=%d, iter %d\n", 
                            i, num_firehawks + j, population_size, iter);
                    goto cleanup;
                }
                float sum = 0.0f;
                for (int d = 0; d < dim; d++) {
                    if (i * dim + d >= population_size * dim || (num_firehawks + j) * dim + d >= population_size * dim) {
                        fprintf(stderr, "Error: population index %zu or %zu out of bounds, max=%d, iter %d\n", 
                                i * dim + d, (num_firehawks + j) * dim + d, population_size * dim, iter);
                        goto cleanup;
                    }
                    float diff = population[i * dim + d] - population[(num_firehawks + j) * dim + d];
                    sum += diff * diff;
                }
                distances[j] = sqrtf(sum);
                sorted_indices[j] = (int)(num_firehawks + j);
                if (sorted_indices[j] < (int)num_firehawks || sorted_indices[j] >= population_size) {
                    fprintf(stderr, "Error: Initial sorted_indices[%zu]=%d out of bounds, iter %d\n", 
                            j, sorted_indices[j], iter);
                    goto cleanup;
                }
            }
            // Sort indices by distance
            for (size_t j = 0; j < prey_size; j++) {
                for (size_t k = j + 1; k < prey_size; k++) {
                    if (j >= population_size || k >= population_size) {
                        fprintf(stderr, "Error: sorted_indices index %zu or %zu out of bounds, max=%d, iter %d\n", 
                                j, k, population_size, iter);
                        goto cleanup;
                    }
                    int idx_j = sorted_indices[j] - num_firehawks;
                    int idx_k = sorted_indices[k] - num_firehawks;
                    if (idx_j < 0 || idx_j >= (int)prey_size || idx_k < 0 || idx_k >= (int)prey_size) {
                        fprintf(stderr, "Error: Sorting indices out of bounds, j=%d, k=%d, iter %d\n", 
                                idx_j, idx_k, iter);
                        goto cleanup;
                    }
                    /*fprintf(stderr, "Iter %d: Sorting j=%zu, k=%zu, idx_j=%d, idx_k=%d, dist_j=%f, dist_k=%f\n",
                            iter, j, k, idx_j, idx_k, distances[idx_j], distances[idx_k]);*/
                    if (distances[idx_j] > distances[idx_k]) {
                        int temp = sorted_indices[j];
                        sorted_indices[j] = sorted_indices[k];
                        sorted_indices[k] = temp;
                    }
                }
            }
            size_t max_prey = population_size - num_firehawks;
            size_t prey_cap = num_firehawks > 0 ? (prey_size + num_firehawks - 1) / num_firehawks : prey_size;
            if (prey_cap > max_prey) prey_cap = max_prey;
            size_t num_prey = max_prey > 0 ? 1 + (rand() % prey_cap) : 1;
            if (num_prey > prey_size) {
                fprintf(stderr, "Error: num_prey %zu exceeds prey_size %zu for firehawk %zu, iter %d\n",
                        num_prey, prey_size, i, iter);
                num_prey = prey_size;
            }
            if (i >= population_size) {
                fprintf(stderr, "Error: prey_counts index %zu out of bounds, max=%d, iter %d\n", 
                        i, population_size, iter);
                goto cleanup;
            }
            prey_counts[i] = (int)num_prey;
            total_prey += num_prey;
            //fprintf(stderr, "Firehawk %zu: num_prey=%zu\n", i, num_prey);
            for (size_t j = 0; j < num_prey; j++) {
                size_t idx = i * prey_size + j;
                if (idx >= max_firehawks * prey_size) {
                    fprintf(stderr, "Error: prey_indices index %zu out of bounds, max=%zu, iter %d\n", 
                            idx, max_firehawks * prey_size, iter);
                    goto cleanup;
                }
                if (j >= population_size) {
                    fprintf(stderr, "Error: sorted_indices index %zu out of bounds, max=%d, iter %d\n", 
                            j, population_size, iter);
                    goto cleanup;
                }
                prey_indices[idx] = sorted_indices[j];
                //fprintf(stderr, "Firehawk %zu: prey_indices[%zu]=%d\n", i, idx, prey_indices[idx]);
                if (prey_indices[idx] < (int)num_firehawks || 
                    prey_indices[idx] >= population_size) {
                    fprintf(stderr, "Error: Invalid prey index %d for firehawk %zu, iter %d\n", 
                            prey_indices[idx], i, iter);
                    goto cleanup;
                }
            }
        }
        if (total_prey > prey_size) {
            fprintf(stderr, "Error: total_prey %zu exceeds prey_size %zu, iter %d\n",
                    total_prey, prey_size, iter);
            goto cleanup;
        }
        fprintf(stderr, "Iter %d: total_prey=%zu\n", iter, total_prey);
        err = clEnqueueWriteBuffer(queue, prey_counts_buffer, CL_TRUE, 0, population_size * sizeof(int), prey_counts, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, prey_indices_buffer, CL_TRUE, 0, max_firehawks * prey_size * sizeof(int), prey_indices, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing prey counts/indices: %d\n", err);
            goto cleanup;
        }

        // Compute safe points
        err = clSetKernelArg(safe_points_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(safe_points_kernel, 1, sizeof(cl_mem), &prey_counts_buffer);
        err |= clSetKernelArg(safe_points_kernel, 2, sizeof(cl_mem), &prey_indices_buffer);
        err |= clSetKernelArg(safe_points_kernel, 3, sizeof(cl_mem), &local_safe_points_buffer);
        err |= clSetKernelArg(safe_points_kernel, 4, sizeof(cl_mem), &global_safe_point_buffer);
        err |= clSetKernelArg(safe_points_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(safe_points_kernel, 6, sizeof(int), &population_size);
        err |= clSetKernelArg(safe_points_kernel, 7, sizeof(int), &num_firehawks);
        err |= clSetKernelArg(safe_points_kernel, 8, sizeof(int), &prey_size);
        err |= clSetKernelArg(safe_points_kernel, 9, dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting safe points kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((num_firehawks + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, safe_points_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing safe points kernel: %d\n", err);
            goto cleanup;
        }
        err = clFinish(queue);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error finishing queue after safe points kernel: %d\n", err);
            goto cleanup;
        }

        // Update firehawks
        err = clSetKernelArg(fh_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(fh_kernel, 1, sizeof(cl_mem), &best_solution_buffer);
        err |= clSetKernelArg(fh_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(fh_kernel, 3, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(fh_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(fh_kernel, 5, sizeof(int), &num_firehawks);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting firehawk kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((num_firehawks + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, fh_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing firehawk kernel: %d\n", err);
            goto cleanup;
        }
        err = clFinish(queue);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error finishing queue after firehawk kernel: %d\n", err);
            goto cleanup;
        }

        // Update prey
        err = clSetKernelArg(prey_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(prey_kernel, 1, sizeof(cl_mem), &prey_counts_buffer);
        err |= clSetKernelArg(prey_kernel, 2, sizeof(cl_mem), &prey_indices_buffer);
        err |= clSetKernelArg(prey_kernel, 3, sizeof(cl_mem), &local_safe_points_buffer);
        err |= clSetKernelArg(prey_kernel, 4, sizeof(cl_mem), &global_safe_point_buffer);
        err |= clSetKernelArg(prey_kernel, 5, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(prey_kernel, 6, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(prey_kernel, 7, sizeof(int), &dim);
        err |= clSetKernelArg(prey_kernel, 8, sizeof(int), &population_size);
        err |= clSetKernelArg(prey_kernel, 9, sizeof(int), &num_firehawks);
        err |= clSetKernelArg(prey_kernel, 10, sizeof(int), &prey_size);
        err |= clSetKernelArg(prey_kernel, 11, sizeof(int), &total_prey);
        err |= clSetKernelArg(prey_kernel, 12, sizeof(float), &select_threshold);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting prey kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((total_prey + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, prey_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[3]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing prey kernel: %d\n", err);
            goto cleanup;
        }
        err = clFinish(queue);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error finishing queue after prey kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate updated population on CPU
        err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading updated population: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < population_size; i++) {
            for (int d = 0; d < dim; d++) {
                if (i * dim + d >= population_size * dim) {
                    fprintf(stderr, "Error: population index %d out of bounds, max=%d\n", 
                            i * dim + d, population_size * dim);
                    goto cleanup;
                }
                cpu_position[d] = (double)population[i * dim + d];
            }
            if (i >= population_size) {
                fprintf(stderr, "Error: fitness index %d out of bounds, max=%d\n", i, population_size);
                goto cleanup;
            }
            fitness[i] = (float)objective_function(cpu_position);
            if (!isfinite(fitness[i])) {
                fprintf(stderr, "Warning: Non-finite fitness for individual %d: %f\n", i, fitness[i]);
                fitness[i] = INFINITY;
            }
        }
        err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing updated fitness: %d\n", err);
            goto cleanup;
        }

        // Find best solution
        err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &prey_indices_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(int), &population_size);
        err |= clSetKernelArg(best_kernel, 3, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 4, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[4]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing best kernel: %d\n", err);
            goto cleanup;
        }
        err = clFinish(queue);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error finishing queue after best kernel: %d\n", err);
            goto cleanup;
        }

        // Update best solution
        int best_idx;
        err = clEnqueueReadBuffer(queue, prey_indices_buffer, CL_TRUE, 0, sizeof(int), &best_idx, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best_idx buffer: %d\n", err);
            goto cleanup;
        }
        fprintf(stderr, "Iter %d: best_idx=%d\n", iter, best_idx);
        if (best_idx >= 0 && best_idx < population_size) {
            float best_fitness_check;
            if (best_idx >= population_size) {
                fprintf(stderr, "Error: best_idx %d out of bounds, max=%d, iter %d\n", 
                        best_idx, population_size, iter);
                goto cleanup;
            }
            err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, best_idx * sizeof(float), sizeof(float), &best_fitness_check, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best fitness: %d\n", err);
                goto cleanup;
            }
            fprintf(stderr, "Iter %d: best_fitness_check=%f, current_best=%f\n", iter, 
                    (double)best_fitness_check, opt->best_solution.fitness);
            if (isfinite(best_fitness_check) && best_fitness_check <= opt->best_solution.fitness) {
                opt->best_solution.fitness = (double)best_fitness_check;
                err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, best_idx * dim * sizeof(float), 
                                          dim * sizeof(float), best_solution, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error reading best solution: %d\n", err);
                    goto cleanup;
                }
                for (int d = 0; d < dim; d++) {
                    if (d >= dim) {
                        fprintf(stderr, "Error: best_solution index %d out of bounds, max=%d, iter %d\n", 
                                d, dim, iter);
                        goto cleanup;
                    }
                    opt->best_solution.position[d] = (double)best_solution[d];
                }
                err = clEnqueueWriteBuffer(queue, best_solution_buffer, CL_TRUE, 0, dim * sizeof(float), best_solution, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error writing best solution: %d\n", err);
                    goto cleanup;
                }
            }
            // Check for stagnation
            if (fabs(best_fitness_check - last_best_fitness) < 1e-4) {
                stagnant_count++;
            } else {
                stagnant_count = 0;
                last_best_fitness = best_fitness_check;
            }
            if (stagnant_count >= 5) {
                fprintf(stderr, "Iter %d: Stagnation detected, reinitializing 75%% of population\n", iter);
                for (int i = population_size / 4; i < population_size; i++) {
                    if (i != best_idx) {
                        for (int d = 0; d < dim; d++) {
                            float min = bounds[2 * d];
                            float max = bounds[2 * d + 1];
                            population[i * dim + d] = min + (max - min) * ((float)rand() / RAND_MAX);
                        }
                        fitness[i] = INFINITY;
                    }
                }
                err = clEnqueueWriteBuffer(queue, population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error writing reinitialized population: %d\n", err);
                    goto cleanup;
                }
                // Evaluate reinitialized population
                err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error reading reinitialized population: %d\n", err);
                    goto cleanup;
                }
                for (int i = 0; i < population_size; i++) {
                    for (int d = 0; d < dim; d++) {
                        cpu_position[d] = (double)population[i * dim + d];
                    }
                    fitness[i] = (float)objective_function(cpu_position);
                    if (!isfinite(fitness[i])) {
                        fprintf(stderr, "Warning: Non-finite fitness for individual %d: %f\n", i, fitness[i]);
                        fitness[i] = INFINITY;
                    }
                }
                err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error writing reinitialized fitness: %d\n", err);
                    goto cleanup;
                }
                stagnant_count = 0;
            }
            // Inject diversity every 5 iterations
            if ((iter + 1) % 5 == 0) {
                fprintf(stderr, "Iter %d: Injecting diversity to non-best individuals\n", iter);
                for (int i = 0; i < population_size; i++) {
                    if (i != best_idx) {
                        for (int d = 0; d < dim; d++) {
                            float perturb = ((float)rand() / RAND_MAX - 0.5f) * 3.0f;
                            population[i * dim + d] += perturb;
                            population[i * dim + d] = fmax(bounds[2 * d], fmin(bounds[2 * d + 1], population[i * dim + d]));
                        }
                        fitness[i] = INFINITY;
                    }
                }
                err = clEnqueueWriteBuffer(queue, population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error writing diversity-injected population: %d\n", err);
                    goto cleanup;
                }
                // Evaluate diversity-injected population
                err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error reading diversity-injected population: %d\n", err);
                    goto cleanup;
                }
                for (int i = 0; i < population_size; i++) {
                    for (int d = 0; d < dim; d++) {
                        cpu_position[d] = (double)population[i * dim + d];
                    }
                    fitness[i] = (float)objective_function(cpu_position);
                    if (!isfinite(fitness[i])) {
                        fprintf(stderr, "Warning: Non-finite fitness for individual %d: %f\n", i, fitness[i]);
                        fitness[i] = INFINITY;
                    }
                }
                err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error writing diversity-injected fitness: %d\n", err);
                    goto cleanup;
                }
            }
        } else {
            fprintf(stderr, "Warning: Invalid best_idx=%d, skipping best solution update, iter %d\n", best_idx, iter);
        }

        // Profiling
        err = clFinish(queue);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error finishing queue at end of iteration: %d\n", err);
            goto cleanup;
        }
        cl_ulong time_start, time_end;
        double init_time = 0.0, safe_points_time = 0.0, fh_time = 0.0, prey_time = 0.0, best_time = 0.0;
        for (int i = 0; i < 5; i++) {
            if (events[i]) {
                err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                err |= clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                if (err == CL_SUCCESS) {
                    double time_ms = (time_end - time_start) / 1e6;
                    if (i == 0) init_time = time_ms;
                    else if (i == 1) safe_points_time = time_ms;
                    else if (i == 2) fh_time = time_ms;
                    else if (i == 3) prey_time = time_ms;
                    else if (i == 4) best_time = time_ms;
                }
                clReleaseEvent(events[i]);
                events[i] = 0;
            }
        }
        end_time = get_time_ms();
        printf("FHO|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | SafePoints: %.3f ms | Firehawks: %.3f ms | Prey: %.3f ms | Best: %.3f ms\n",
               iter + 1, opt->best_solution.fitness, end_time - start_time, 
               init_time, safe_points_time, fh_time, prey_time, best_time);
    }

    // Update CPU-side population
    err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final buffers: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            if (i * dim + d >= population_size * dim) {
                fprintf(stderr, "Error: population index %d out of bounds, max=%d\n", 
                        i * dim + d, population_size * dim);
                goto cleanup;
            }
            float p = population[i * dim + d];
            float lower_bound = bounds[d * 2];
            float upper_bound = bounds[d * 2 + 1];
            if (p < lower_bound || p > upper_bound || !isfinite(p)) {
                fprintf(stderr, "Warning: Final position out of bounds or non-finite for individual %d, dim %d: %f\n", i, d, p);
                p = lower_bound + (upper_bound - lower_bound) * ((float)rand() / RAND_MAX);
                population[i * dim + d] = p;
            }
            opt->population[i].position[d] = (double)p;
        }
        if (i >= population_size) {
            fprintf(stderr, "Error: fitness index %d out of bounds, max=%d\n", i, population_size);
            goto cleanup;
        }
        opt->population[i].fitness = (double)fitness[i];
    }

cleanup:
    // Safely free host memory
    fprintf(stderr, "Freeing host memory: population=%p, fitness=%p, bounds=%p, best_solution=%p, prey_counts=%p, prey_indices=%p, rng_states=%p, cpu_position=%p, distances=%p, sorted_indices=%p\n",
            (void*)population, (void*)fitness, (void*)bounds, (void*)best_solution, (void*)prey_counts, 
            (void*)prey_indices, (void*)rng_states, (void*)cpu_position, (void*)distances, (void*)sorted_indices);
    if (population) { free(population); population = NULL; }
    if (fitness) { free(fitness); fitness = NULL; }
    if (bounds) { free(bounds); bounds = NULL; }
    if (best_solution) { free(best_solution); best_solution = NULL; }
    if (prey_counts) { free(prey_counts); prey_counts = NULL; }
    if (prey_indices) { free(prey_indices); prey_indices = NULL; }
    if (rng_states) { free(rng_states); rng_states = NULL; }
    if (cpu_position) { free(cpu_position); cpu_position = NULL; }
    if (distances) { free(distances); distances = NULL; }
    if (sorted_indices) { free(sorted_indices); sorted_indices = NULL; }
    // Safely release OpenCL resources
    fprintf(stderr, "Releasing OpenCL resources\n");
    if (population_buffer) { clReleaseMemObject(population_buffer); population_buffer = NULL; }
    if (fitness_buffer) { clReleaseMemObject(fitness_buffer); fitness_buffer = NULL; }
    if (bounds_buffer) { clReleaseMemObject(bounds_buffer); bounds_buffer = NULL; }
    if (best_solution_buffer) { clReleaseMemObject(best_solution_buffer); best_solution_buffer = NULL; }
    if (prey_counts_buffer) { clReleaseMemObject(prey_counts_buffer); prey_counts_buffer = NULL; }
    if (prey_indices_buffer) { clReleaseMemObject(prey_indices_buffer); prey_indices_buffer = NULL; }
    if (local_safe_points_buffer) { clReleaseMemObject(local_safe_points_buffer); local_safe_points_buffer = NULL; }
    if (global_safe_point_buffer) { clReleaseMemObject(global_safe_point_buffer); global_safe_point_buffer = NULL; }
    if (rng_states_buffer) { clReleaseMemObject(rng_states_buffer); rng_states_buffer = NULL; }
    if (init_kernel) { clReleaseKernel(init_kernel); init_kernel = NULL; }
    if (fh_kernel) { clReleaseKernel(fh_kernel); fh_kernel = NULL; }
    if (safe_points_kernel) { clReleaseKernel(safe_points_kernel); safe_points_kernel = NULL; }
    if (prey_kernel) { clReleaseKernel(prey_kernel); prey_kernel = NULL; }
    if (best_kernel) { clReleaseKernel(best_kernel); best_kernel = NULL; }
    if (program) { clReleaseProgram(program); program = NULL; }
    for (int i = 0; i < 5; i++) {
        if (events[i]) { clReleaseEvent(events[i]); events[i] = 0; }
    }
    if (queue) {
        clFinish(queue);
        clReleaseCommandQueue(queue);
        queue = NULL;
    }
    fprintf(stderr, "Cleanup complete\n");
}
