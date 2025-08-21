#include "CSO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <float.h>
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

// Random permutation generator (CPU-based)
static void randperm_f(int range_size, int dim, int *result, int *temp) {
    for (int i = 0; i < range_size; i++) {
        temp[i] = i + 1;
    }
    for (int i = range_size - 1; i > 0; i--) {
        int j = (int)(((double)rand() / RAND_MAX) * (i + 1));
        int swap = temp[i];
        temp[i] = temp[j];
        temp[j] = swap;
    }
    for (int i = 0; i < dim; i++) {
        result[i] = (i < range_size) ? temp[i] : (1 + (int)(((double)rand() / RAND_MAX) * range_size));
    }
}

// Comparison function for qsort (CPU-based sorting)
static float *g_fitness = NULL;
static int compare_fitness(const void *a, const void *b) {
    int idx_a = *(int *)a;
    int idx_b = *(int *)b;
    return (g_fitness[idx_a] > g_fitness[idx_b]) - (g_fitness[idx_a] < g_fitness[idx_b]);
}

// OpenCL kernel source for CSO
static const char *cso_kernel_source =
"#define ROOSTER_RATIO 0.15f\n"
"#define HEN_RATIO 0.7f\n"
"#define MOTHER_RATIO 0.5f\n"
// Random number generator
"uint lcg_rand(uint *seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float rand_float(uint *seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
// Initialize kernel
"__kernel void initialize_cso(\n"
"    __global float *positions,\n"
"    __global float *fitness,\n"
"    __global uint *rng_states,\n"
"    __global const float *bounds,\n"
"    const int dim,\n"
"    const int pop_size)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= pop_size) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float lower = bounds[d * 2];\n"
"        float upper = bounds[d * 2 + 1];\n"
"        positions[idx * dim + d] = lower + (upper - lower) * rand_float(&rng_state, 0.0f, 1.0f);\n"
"    }\n"
"    fitness[idx] = INFINITY;\n"
"    rng_states[idx] = rng_state;\n"
"}\n"
// Update roosters kernel
"__kernel void update_roosters(\n"
"    __global float *positions,\n"
"    __global float *fitness,\n"
"    __global int *sort_indices,\n"
"    __global uint *rng_states,\n"
"    __global const float *bounds,\n"
"    __global float *new_positions,\n"
"    const int dim,\n"
"    const int rooster_num)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= rooster_num) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    int curr_idx = sort_indices[idx];\n"
"    int another_rooster = (int)(rand_float(&rng_state, 0.0f, (float)rooster_num));\n"
"    if (another_rooster == idx) another_rooster = (another_rooster + 1) % rooster_num;\n"
"    int another_idx = sort_indices[another_rooster];\n"
"    float sigma = 1.0f;\n"
"    float curr_fitness = fitness[curr_idx];\n"
"    float other_fitness = fitness[another_idx];\n"
"    if (isfinite(curr_fitness) && isfinite(other_fitness) && curr_fitness > other_fitness) {\n"
"        float diff = other_fitness - curr_fitness;\n"
"        float denom = fabs(curr_fitness) + 2.2e-16f;\n"
"        if (fabs(diff / denom) < 100.0f) {\n"
"            sigma = exp(diff / denom);\n"
"        }\n"
"    }\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float factor = 1.0f + sigma * rand_float(&rng_state, -3.0f, 3.0f);\n"
"        float new_pos = positions[curr_idx * dim + d] * factor;\n"
"        float lower = bounds[d * 2];\n"
"        float upper = bounds[d * 2 + 1];\n"
"        new_pos = fmin(fmax(new_pos, lower), upper);\n"
"        new_positions[idx * dim + d] = new_pos;\n"
"    }\n"
"    rng_states[idx] = rng_state;\n"
"}\n"
// Update hens kernel
"__kernel void update_hens(\n"
"    __global float *positions,\n"
"    __global float *fitness,\n"
"    __global int *sort_indices,\n"
"    __global int *mate,\n"
"    __global uint *rng_states,\n"
"    __global const float *bounds,\n"
"    __global float *new_positions,\n"
"    const int dim,\n"
"    const int rooster_num,\n"
"    const int hen_num)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= hen_num) return;\n"
"    uint rng_state = rng_states[idx + rooster_num];\n"
"    int curr_idx = sort_indices[idx + rooster_num];\n"
"    int mate_idx = sort_indices[mate[idx] - 1];\n"
"    int other = (int)(rand_float(&rng_state, 0.0f, (float)(rooster_num + idx)));\n"
"    if (other == mate[idx] - 1) other = (other + 1) % (rooster_num + idx);\n"
"    int other_idx = sort_indices[other];\n"
"    float c1 = 1.0f, c2 = 1.0f;\n"
"    float curr_fitness = fitness[curr_idx];\n"
"    float mate_fitness = fitness[mate_idx];\n"
"    float other_fitness = fitness[other_idx];\n"
"    if (isfinite(curr_fitness) && isfinite(mate_fitness) && isfinite(other_fitness)) {\n"
"        float diff1 = curr_fitness - mate_fitness;\n"
"        float denom1 = fabs(curr_fitness) + 2.2e-16f;\n"
"        if (fabs(diff1 / denom1) < 100.0f) {\n"
"            c1 = exp(diff1 / denom1);\n"
"        }\n"
"        float diff2 = other_fitness - curr_fitness;\n"
"        if (fabs(diff2) < 100.0f) {\n"
"            c2 = exp(diff2);\n"
"        }\n"
"    }\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float delta1 = c1 * rand_float(&rng_state, 0.0f, 1.0f) * (positions[mate_idx * dim + d] - positions[curr_idx * dim + d]);\n"
"        float delta2 = c2 * rand_float(&rng_state, 0.0f, 1.0f) * (positions[other_idx * dim + d] - positions[curr_idx * dim + d]);\n"
"        float new_pos = positions[curr_idx * dim + d] + delta1 + delta2;\n"
"        float lower = bounds[d * 2];\n"
"        float upper = bounds[d * 2 + 1];\n"
"        new_pos = fmin(fmax(new_pos, lower), upper);\n"
"        new_positions[idx * dim + d] = new_pos;\n"
"    }\n"
"    rng_states[idx + rooster_num] = rng_state;\n"
"}\n"
// Update chicks kernel
"__kernel void update_chicks(\n"
"    __global float *positions,\n"
"    __global int *sort_indices,\n"
"    __global int *mother_lib,\n"
"    __global uint *rng_states,\n"
"    __global const float *bounds,\n"
"    __global float *new_positions,\n"
"    const int dim,\n"
"    const int rooster_num,\n"
"    const int hen_num,\n"
"    const int mother_num)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int chick_num = get_global_size(0);\n"
"    if (idx >= chick_num) return;\n"
"    uint rng_state = rng_states[idx + rooster_num + hen_num];\n"
"    int curr_idx = sort_indices[idx + rooster_num + hen_num];\n"
"    int mother_idx = sort_indices[mother_lib[(int)(rand_float(&rng_state, 0.0f, (float)mother_num))] - 1];\n"
"    float fl = 0.5f + 0.4f * rand_float(&rng_state, 0.0f, 1.0f);\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float delta = fl * (positions[mother_idx * dim + d] - positions[curr_idx * dim + d]);\n"
"        float new_pos = positions[curr_idx * dim + d] + delta;\n"
"        float lower = bounds[d * 2];\n"
"        float upper = bounds[d * 2 + 1];\n"
"        new_pos = fmin(fmax(new_pos, lower), upper);\n"
"        new_positions[idx * dim + d] = new_pos;\n"
"    }\n"
"    rng_states[idx + rooster_num + hen_num] = rng_state;\n"
"}\n"
// Find best solution
"__kernel void find_best(\n"
"    __global const float *fitness,\n"
"    __global int *best_idx,\n"
"    const int pop_size,\n"
"    __local float *local_fitness,\n"
"    __local int *local_indices)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    if (idx < pop_size) {\n"
"        local_fitness[local_id] = fitness[idx];\n"
"        local_indices[local_id] = idx;\n"
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
void CSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, rooster_kernel = NULL, hen_kernel = NULL;
    cl_kernel chick_kernel = NULL, best_kernel = NULL;
    cl_command_queue queue = NULL;
    cl_mem positions_buffer = NULL, fitness_buffer = NULL, sort_indices_buffer = NULL;
    cl_mem mate_buffer = NULL, mother_lib_buffer = NULL, rng_states_buffer = NULL;
    cl_mem bounds_buffer = NULL, new_positions_buffer = NULL, best_idx_buffer = NULL;
    float *positions = NULL, *fitness = NULL, *new_positions = NULL, *bounds = NULL;
    int *sort_indices = NULL, *mate = NULL, *mother_lib = NULL, *temp_perm = NULL;
    uint *rng_states = NULL;
    double *cpu_position = NULL;
    cl_event events[5] = {0};
    double start_time, end_time;

    // Validate input
    if (!opt || !opt->population || !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure, null pointers, or missing objective function\n");
        goto cleanup;
    }
    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    const int max_iter = opt->max_iter;
    if (dim < 1 || pop_size < 1 || max_iter < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), pop_size (%d), or max_iter (%d)\n", 
                dim, pop_size, max_iter);
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

    // Pre-compute constants
    const int rooster_num = (int)(pop_size * ROOSTER_RATIO);
    const int hen_num = (int)(pop_size * HEN_RATIO);
    const int mother_num = (int)(hen_num * MOTHER_RATIO);
    const int chick_num = pop_size - rooster_num - hen_num;
    if (rooster_num < 1 || hen_num < 1 || chick_num < 1 || mother_num < 1) {
        fprintf(stderr, "Error: Invalid group sizes: roosters=%d, hens=%d, chicks=%d, mothers=%d\n",
                rooster_num, hen_num, chick_num, mother_num);
        goto cleanup;
    }

    // Initialize OpenCL program
    program = clCreateProgramWithSource(opt->context, 1, &cso_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating CSO program: %d\n", err);
        goto cleanup;
    }
    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building CSO program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        goto cleanup;
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_cso", &err);
    rooster_kernel = clCreateKernel(program, "update_roosters", &err);
    hen_kernel = clCreateKernel(program, "update_hens", &err);
    chick_kernel = clCreateKernel(program, "update_chicks", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating CSO kernels: %d\n", err);
        goto cleanup;
    }

    // Create command queue with profiling
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue = clCreateCommandQueueWithProperties(opt->context, opt->device, props, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating command queue: %d\n", err);
        goto cleanup;
    }

    // Set work-group size
    const size_t local_work_size = 32; // Hardcoded for Intel GPU stability

    // Allocate GPU buffers
    positions_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(float), NULL, &err);
    sort_indices_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(int), NULL, &err);
    mate_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, hen_num * sizeof(int), NULL, &err);
    mother_lib_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, mother_num * sizeof(int), NULL, &err);
    rng_states_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(uint), NULL, &err);
    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, dim * 2 * sizeof(float), NULL, &err);
    new_positions_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating CSO buffers: %d\n", err);
        goto cleanup;
    }

    // Allocate host memory
    positions = (float *)malloc(pop_size * dim * sizeof(float));
    fitness = (float *)malloc(pop_size * sizeof(float));
    sort_indices = (int *)malloc(pop_size * sizeof(int));
    mate = (int *)malloc(hen_num * sizeof(int));
    mother_lib = (int *)malloc(mother_num * sizeof(int));
    temp_perm = (int *)malloc(pop_size * sizeof(int));
    bounds = (float *)malloc(dim * 2 * sizeof(float));
    new_positions = (float *)malloc(pop_size * dim * sizeof(float));
    rng_states = (uint *)malloc(pop_size * sizeof(uint));
    cpu_position = (double *)malloc(dim * sizeof(double));
    if (!positions || !fitness || !sort_indices || !mate || !mother_lib || !temp_perm ||
        !bounds || !new_positions || !rng_states || !cpu_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }

    // Initialize host data
    srand((unsigned int)time(NULL));
    for (int d = 0; d < dim; d++) {
        bounds[d * 2] = (float)opt->bounds[d * 2];
        bounds[d * 2 + 1] = (float)opt->bounds[d * 2 + 1];
    }
    for (int i = 0; i < pop_size; i++) {
        rng_states[i] = (uint)(time(NULL) ^ (i + 1));
        sort_indices[i] = i;
        for (int d = 0; d < dim; d++) {
            positions[i * dim + d] = (float)opt->population[i].position[d];
        }
        fitness[i] = INFINITY;
    }
    for (int i = 0; i < hen_num; i++) {
        mate[i] = 0;
    }
    for (int i = 0; i < mother_num; i++) {
        mother_lib[i] = 0;
    }

    // Write initial data to GPU
    err = clEnqueueWriteBuffer(queue, positions_buffer, CL_TRUE, 0, 
                              pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(float), fitness, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, sort_indices_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(int), sort_indices, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, mate_buffer, CL_TRUE, 0, 
                               hen_num * sizeof(int), mate, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, mother_lib_buffer, CL_TRUE, 0, 
                               mother_num * sizeof(int), mother_lib, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, rng_states_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(uint), rng_states, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 
                               dim * 2 * sizeof(float), bounds, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial buffers: %d\n", err);
        goto cleanup;
    }

    // Initialize population
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &positions_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &fitness_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(cl_mem), &rng_states_buffer);
    err |= clSetKernelArg(init_kernel, 3, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 4, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 5, sizeof(int), &pop_size);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting init kernel args: %d\n", err);
        goto cleanup;
    }
    size_t global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[0]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel: %d\n", err);
        goto cleanup;
    }

    // Evaluate initial fitness on CPU
    err = clEnqueueReadBuffer(queue, positions_buffer, CL_TRUE, 0, 
                             pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading initial positions: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < pop_size; i++) {
        for (int d = 0; d < dim; d++) {
            cpu_position[d] = (double)positions[i * dim + d];
        }
        float new_fitness = (float)objective_function(cpu_position);
        if (!isfinite(new_fitness)) {
            fprintf(stderr, "Warning: Non-finite fitness for chicken %d: %f\n", i, new_fitness);
            new_fitness = INFINITY;
        }
        fitness[i] = new_fitness;
    }
    err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                              pop_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial fitness: %d\n", err);
        goto cleanup;
    }

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < pop_size; i++) {
        if (fitness[i] < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)fitness[i];
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)positions[i * dim + d];
            }
        }
    }

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Update sort indices and relationships
        if (iter % UPDATE_FREQ == 0 || iter == 0) {
            // CPU-based sorting
            for (int i = 0; i < pop_size; i++) {
                sort_indices[i] = i;
            }
            g_fitness = fitness;
            qsort(sort_indices, pop_size, sizeof(int), compare_fitness);
            g_fitness = NULL;
            err = clEnqueueWriteBuffer(queue, sort_indices_buffer, CL_TRUE, 0, 
                                      pop_size * sizeof(int), sort_indices, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing sort_indices: %d\n", err);
                goto cleanup;
            }

            // Generate mate and mother assignments
            randperm_f(rooster_num, hen_num, mate, temp_perm);
            randperm_f(hen_num, mother_num, mother_lib, temp_perm);
            for (int i = 0; i < mother_num; i++) {
                mother_lib[i] += rooster_num;
            }
            err = clEnqueueWriteBuffer(queue, mate_buffer, CL_TRUE, 0, 
                                      hen_num * sizeof(int), mate, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(queue, mother_lib_buffer, CL_TRUE, 0, 
                                       mother_num * sizeof(int), mother_lib, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing mate/mother_lib: %d\n", err);
                goto cleanup;
            }
        }

        // Update roosters
        err = clSetKernelArg(rooster_kernel, 0, sizeof(cl_mem), &positions_buffer);
        err |= clSetKernelArg(rooster_kernel, 1, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(rooster_kernel, 2, sizeof(cl_mem), &sort_indices_buffer);
        err |= clSetKernelArg(rooster_kernel, 3, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(rooster_kernel, 4, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(rooster_kernel, 5, sizeof(cl_mem), &new_positions_buffer);
        err |= clSetKernelArg(rooster_kernel, 6, sizeof(int), &dim);
        err |= clSetKernelArg(rooster_kernel, 7, sizeof(int), &rooster_num);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting rooster kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((rooster_num + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, rooster_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing rooster kernel: %d\n", err);
            goto cleanup;
        }

        // Update hens
        err = clSetKernelArg(hen_kernel, 0, sizeof(cl_mem), &positions_buffer);
        err |= clSetKernelArg(hen_kernel, 1, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(hen_kernel, 2, sizeof(cl_mem), &sort_indices_buffer);
        err |= clSetKernelArg(hen_kernel, 3, sizeof(cl_mem), &mate_buffer);
        err |= clSetKernelArg(hen_kernel, 4, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(hen_kernel, 5, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(hen_kernel, 6, sizeof(cl_mem), &new_positions_buffer);
        err |= clSetKernelArg(hen_kernel, 7, sizeof(int), &dim);
        err |= clSetKernelArg(hen_kernel, 8, sizeof(int), &rooster_num);
        err |= clSetKernelArg(hen_kernel, 9, sizeof(int), &hen_num);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting hen kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((hen_num + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, hen_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing hen kernel: %d\n", err);
            goto cleanup;
        }

        // Update chicks
        err = clSetKernelArg(chick_kernel, 0, sizeof(cl_mem), &positions_buffer);
        err |= clSetKernelArg(chick_kernel, 1, sizeof(cl_mem), &sort_indices_buffer);
        err |= clSetKernelArg(chick_kernel, 2, sizeof(cl_mem), &mother_lib_buffer);
        err |= clSetKernelArg(chick_kernel, 3, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(chick_kernel, 4, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(chick_kernel, 5, sizeof(cl_mem), &new_positions_buffer);
        err |= clSetKernelArg(chick_kernel, 6, sizeof(int), &dim);
        err |= clSetKernelArg(chick_kernel, 7, sizeof(int), &rooster_num);
        err |= clSetKernelArg(chick_kernel, 8, sizeof(int), &hen_num);
        err |= clSetKernelArg(chick_kernel, 9, sizeof(int), &mother_num);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting chick kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((chick_num + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, chick_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[3]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing chick kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate new positions on CPU
        err = clEnqueueReadBuffer(queue, new_positions_buffer, CL_TRUE, 0, 
                                 pop_size * dim * sizeof(float), new_positions, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(queue, sort_indices_buffer, CL_TRUE, 0, 
                                  pop_size * sizeof(int), sort_indices, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading new_positions/sort_indices: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < pop_size; i++) {
            int curr_idx = sort_indices[i];
            for (int d = 0; d < dim; d++) {
                cpu_position[d] = (double)new_positions[i * dim + d];
            }
            float new_fitness = (float)objective_function(cpu_position);
            if (!isfinite(new_fitness)) {
                fprintf(stderr, "Warning: Non-finite fitness for chicken %d: %f\n", curr_idx, new_fitness);
                new_fitness = INFINITY;
            }
            fitness[curr_idx] = new_fitness;
            for (int d = 0; d < dim; d++) {
                positions[curr_idx * dim + d] = new_positions[i * dim + d];
            }
        }

        // Write updated data to GPU
        err = clEnqueueWriteBuffer(queue, positions_buffer, CL_FALSE, 0, 
                                  pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                                   pop_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing positions/fitness: %d\n", err);
            goto cleanup;
        }

        // Find best solution
        err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &best_idx_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(int), &pop_size);
        err |= clSetKernelArg(best_kernel, 3, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 4, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
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
        if (best_idx >= 0 && best_idx < pop_size) {
            float best_fitness_check;
            err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, best_idx * sizeof(float), 
                                     sizeof(float), &best_fitness_check, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best fitness: %d\n", err);
                goto cleanup;
            }
            if (isfinite(best_fitness_check) && best_fitness_check < opt->best_solution.fitness) {
                opt->best_solution.fitness = (double)best_fitness_check;
                float *best_solution = (float *)malloc(dim * sizeof(float));
                if (!best_solution) {
                    fprintf(stderr, "Error: Memory allocation failed for best_solution\n");
                    goto cleanup;
                }
                err = clEnqueueReadBuffer(queue, positions_buffer, CL_TRUE, best_idx * dim * sizeof(float), 
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
        } else {
            fprintf(stderr, "Warning: Invalid best_idx %d at iteration %d\n", best_idx, iter + 1);
        }

        // Flush and finish queue
        clFlush(queue);
        err = clFinish(queue);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error finishing queue: %d\n", err);
            goto cleanup;
        }

        // Profiling
        cl_ulong time_start, time_end;
        double init_time = 0.0, rooster_time = 0.0, hen_time = 0.0, chick_time = 0.0, best_time = 0.0;
        for (int i = 0; i < 5; i++) {
            if (events[i]) {
                err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                err |= clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                if (err == CL_SUCCESS) {
                    double time_ms = (time_end - time_start) / 1e6;
                    if (i == 0) init_time = time_ms;
                    else if (i == 1) rooster_time = time_ms;
                    else if (i == 2) hen_time = time_ms;
                    else if (i == 3) chick_time = time_ms;
                    else if (i == 4) best_time = time_ms;
                }
            }
        }
        end_time = get_time_ms();
        printf("CSO|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Rooster: %.3f ms | Hen: %.3f ms | Chick: %.3f ms | Best: %.3f ms\n",
               iter + 1, opt->best_solution.fitness, end_time - start_time, 
               init_time, rooster_time, hen_time, chick_time, best_time);

        // Reset events
        for (int i = 0; i < 5; i++) {
            if (events[i]) {
                clReleaseEvent(events[i]);
                events[i] = 0;
            }
        }
    }

    // Update CPU-side population
    err = clEnqueueReadBuffer(queue, positions_buffer, CL_TRUE, 0, 
                             pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                              pop_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final buffers: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < pop_size; i++) {
        for (int d = 0; d < dim; d++) {
            float p = positions[i * dim + d];
            float lower_bound = bounds[d * 2];
            float upper_bound = bounds[d * 2 + 1];
            if (p < lower_bound || p > upper_bound || !isfinite(p)) {
                fprintf(stderr, "Warning: Final position out of bounds or non-finite for chicken %d, dim %d: %f\n", i, d, p);
                p = lower_bound + (upper_bound - lower_bound) * ((float)rand() / RAND_MAX);
                positions[i * dim + d] = p;
            }
            opt->population[i].position[d] = (double)p;
        }
        opt->population[i].fitness = (double)fitness[i];
    }

cleanup:
    if (positions) free(positions);
    if (fitness) free(fitness);
    if (sort_indices) free(sort_indices);
    if (mate) free(mate);
    if (mother_lib) free(mother_lib);
    if (temp_perm) free(temp_perm);
    if (bounds) free(bounds);
    if (new_positions) free(new_positions);
    if (rng_states) free(rng_states);
    if (cpu_position) free(cpu_position);
    if (positions_buffer) clReleaseMemObject(positions_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (sort_indices_buffer) clReleaseMemObject(sort_indices_buffer);
    if (mate_buffer) clReleaseMemObject(mate_buffer);
    if (mother_lib_buffer) clReleaseMemObject(mother_lib_buffer);
    if (rng_states_buffer) clReleaseMemObject(rng_states_buffer);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (new_positions_buffer) clReleaseMemObject(new_positions_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (rooster_kernel) clReleaseKernel(rooster_kernel);
    if (hen_kernel) clReleaseKernel(hen_kernel);
    if (chick_kernel) clReleaseKernel(chick_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 5; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) {
        clFinish(queue);
        clReleaseCommandQueue(queue);
    }
}
