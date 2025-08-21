#include "CO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
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

// OpenCL kernel source for CO
static const char *co_kernel_source =
"#define MIN_EGGS 2\n"
"#define MAX_EGGS 4\n"
"#define MAX_CUCKOOS 10\n"
"#define MAX_EGGS_PER_CUCKOO 4\n"
"#define RADIUS_COEFF 5.0f\n"
"#define MOTION_COEFF 9.0f\n"
"#define PI 3.14159265358979323846f\n"
// Random number generator
"uint lcg_rand(uint *seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float rand_float(uint *seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
// Initialize cuckoos kernel
"__kernel void initialize_cuckoos(\n"
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
// Lay eggs kernel
"__kernel void lay_eggs(\n"
"    __global float *cuckoo_positions,\n"
"    __global float *egg_positions,\n"
"    __global int *num_eggs,\n"
"    __global uint *rng_states,\n"
"    __global const float *bounds,\n"
"    __global const float *rand_buffer,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    const int total_eggs)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= pop_size) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    int eggs = num_eggs[idx];\n"
"    int egg_offset = 0;\n"
"    for (int i = 0; i < idx; i++) egg_offset += num_eggs[i];\n"
"    float radius_factor = ((float)eggs / total_eggs) * RADIUS_COEFF;\n"
"    int rand_idx = pop_size + egg_offset * 2;\n"
"    for (int k = 0; k < eggs; k++) {\n"
"        float random_scalar = rand_buffer[rand_idx++];\n"
"        float angle = k * (2 * PI / eggs);\n"
"        float sign = (rand_buffer[rand_idx++] < 0.5f) ? 1.0f : -1.0f;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float bound_range = bounds[d * 2 + 1] - bounds[d * 2];\n"
"            float radius = radius_factor * random_scalar * bound_range;\n"
"            float adding_value = sign * radius * cos(angle) + radius * sin(angle);\n"
"            float new_pos = cuckoo_positions[idx * dim + d] + adding_value;\n"
"            new_pos = fmin(fmax(new_pos, bounds[d * 2]), bounds[d * 2 + 1]);\n"
"            egg_positions[(egg_offset + k) * dim + d] = new_pos;\n"
"        }\n"
"    }\n"
"    rng_states[idx] = rng_state;\n"
"}\n"
// Select best cuckoos kernel
"__kernel void select_best(\n"
"    __global float *all_positions,\n"
"    __global float *all_fitness,\n"
"    __global float *new_positions,\n"
"    __global float *new_fitness,\n"
"    const int dim,\n"
"    const int total_positions,\n"
"    const int max_cuckoos,\n"
"    __local float *local_fitness,\n"
"    __local int *local_indices)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    if (idx < total_positions) {\n"
"        local_fitness[local_id] = all_fitness[idx];\n"
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
"    if (local_id == 0 && get_group_id(0) < max_cuckoos) {\n"
"        int selected_idx = local_indices[0];\n"
"        if (selected_idx >= 0) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                new_positions[get_group_id(0) * dim + d] = all_positions[selected_idx * dim + d];\n"
"            }\n"
"            new_fitness[get_group_id(0)] = all_fitness[selected_idx];\n"
"        }\n"
"    }\n"
"}\n"
// Cluster and migrate kernel
"__kernel void cluster_and_migrate(\n"
"    __global float *positions,\n"
"    __global float *fitness,\n"
"    __global uint *rng_states,\n"
"    __global const float *bounds,\n"
"    __global float *new_positions,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    const int best_idx)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= pop_size) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float delta = positions[best_idx * dim + d] - positions[idx * dim + d];\n"
"        float new_pos = positions[idx * dim + d] + MOTION_COEFF * rand_float(&rng_state, 0.0f, 1.0f) * delta;\n"
"        new_pos = fmin(fmax(new_pos, bounds[d * 2]), bounds[d * 2 + 1]);\n"
"        new_positions[idx * dim + d] = new_pos;\n"
"    }\n"
"    rng_states[idx] = rng_state;\n"
"}\n";

// Main Optimization Function
void CO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, egg_kernel = NULL, select_kernel = NULL, migrate_kernel = NULL;
    cl_command_queue queue = NULL;
    cl_mem cuckoo_positions_buffer = NULL, egg_positions_buffer = NULL, all_positions_buffer = NULL;
    cl_mem all_fitness_buffer = NULL, new_positions_buffer = NULL, new_fitness_buffer = NULL;
    cl_mem num_eggs_buffer = NULL, rng_states_buffer = NULL, bounds_buffer = NULL, rand_buffer = NULL;
    float *cuckoo_positions = NULL, *egg_positions = NULL, *all_positions = NULL, *all_fitness = NULL;
    float *new_positions = NULL, *new_fitness = NULL, *bounds = NULL, *rand_buffer_host = NULL;
    int *num_eggs = NULL;
    uint *rng_states = NULL;
    double *cpu_position = NULL;
    cl_event events[4] = {0};
    double start_time, end_time;

    // Validate input
    if (!opt || !opt->population || !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure, null pointers, or missing objective function\n");
        goto cleanup;
    }
    const int dim = opt->dim;
    int pop_size = opt->population_size;
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
    if (pop_size > MAX_CUCKOOS) {
        pop_size = MAX_CUCKOOS;
        opt->population_size = MAX_CUCKOOS;
        printf("Warning: pop_size reduced to MAX_CUCKOOS (%d)\n", MAX_CUCKOOS);
    }
    for (int d = 0; d < dim; d++) {
        if (opt->bounds[d * 2] >= opt->bounds[d * 2 + 1]) {
            fprintf(stderr, "Error: Invalid bounds for dimension %d: [%f, %f]\n", 
                    d, opt->bounds[d * 2], opt->bounds[d * 2 + 1]);
            goto cleanup;
        }
    }

    // Pre-compute constants
    const int max_total_eggs = pop_size * MAX_EGGS_PER_CUCKOO;
    const int max_total_positions = pop_size + max_total_eggs;

    // Initialize OpenCL program
    program = clCreateProgramWithSource(opt->context, 1, &co_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating CO program: %d\n", err);
        goto cleanup;
    }
    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building CO program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        goto cleanup;
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_cuckoos", &err);
    egg_kernel = clCreateKernel(program, "lay_eggs", &err);
    select_kernel = clCreateKernel(program, "select_best", &err);
    migrate_kernel = clCreateKernel(program, "cluster_and_migrate", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating CO kernels: %d\n", err);
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
    cuckoo_positions_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    egg_positions_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, max_total_eggs * dim * sizeof(float), NULL, &err);
    all_positions_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, max_total_positions * dim * sizeof(float), NULL, &err);
    all_fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, max_total_positions * sizeof(float), NULL, &err);
    new_positions_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, MAX_CUCKOOS * dim * sizeof(float), NULL, &err);
    new_fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, MAX_CUCKOOS * sizeof(float), NULL, &err);
    num_eggs_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(int), NULL, &err);
    rng_states_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(uint), NULL, &err);
    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, dim * 2 * sizeof(float), NULL, &err);
    rand_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, (pop_size + max_total_eggs * 2) * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating CO buffers: %d\n", err);
        goto cleanup;
    }

    // Allocate host memory
    cuckoo_positions = (float *)malloc(pop_size * dim * sizeof(float));
    egg_positions = (float *)malloc(max_total_eggs * dim * sizeof(float));
    all_positions = (float *)malloc(max_total_positions * dim * sizeof(float));
    all_fitness = (float *)malloc(max_total_positions * sizeof(float));
    new_positions = (float *)malloc(MAX_CUCKOOS * dim * sizeof(float));
    new_fitness = (float *)malloc(MAX_CUCKOOS * sizeof(float));
    num_eggs = (int *)malloc(pop_size * sizeof(int));
    rng_states = (uint *)malloc(pop_size * sizeof(uint));
    bounds = (float *)malloc(dim * 2 * sizeof(float));
    rand_buffer_host = (float *)malloc((pop_size + max_total_eggs * 2) * sizeof(float));
    cpu_position = (double *)malloc(dim * sizeof(double));
    if (!cuckoo_positions || !egg_positions || !all_positions || !all_fitness || !new_positions ||
        !new_fitness || !num_eggs || !rng_states || !bounds || !rand_buffer_host || !cpu_position) {
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
        for (int d = 0; d < dim; d++) {
            cuckoo_positions[i * dim + d] = (float)opt->population[i].position[d];
        }
    }

    // Write initial data to GPU
    err = clEnqueueWriteBuffer(queue, cuckoo_positions_buffer, CL_TRUE, 0, 
                              pop_size * dim * sizeof(float), cuckoo_positions, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, rng_states_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(uint), rng_states, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 
                               dim * 2 * sizeof(float), bounds, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial buffers: %d\n", err);
        goto cleanup;
    }

    // Initialize cuckoos
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &cuckoo_positions_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &all_fitness_buffer);
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
    err = clEnqueueReadBuffer(queue, cuckoo_positions_buffer, CL_TRUE, 0, 
                             pop_size * dim * sizeof(float), cuckoo_positions, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading initial positions: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < pop_size; i++) {
        for (int d = 0; d < dim; d++) {
            cpu_position[d] = (double)cuckoo_positions[i * dim + d];
        }
        float new_fitness = (float)objective_function(cpu_position);
        if (!isfinite(new_fitness)) {
            fprintf(stderr, "Warning: Non-finite fitness for cuckoo %d: %f\n", i, new_fitness);
            new_fitness = INFINITY;
        }
        all_fitness[i] = new_fitness;
    }
    err = clEnqueueWriteBuffer(queue, all_fitness_buffer, CL_TRUE, 0, 
                              pop_size * sizeof(float), all_fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial fitness: %d\n", err);
        goto cleanup;
    }

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < pop_size; i++) {
        if (all_fitness[i] < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)all_fitness[i];
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)cuckoo_positions[i * dim + d];
            }
        }
    }

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Generate random numbers
        for (int i = 0; i < pop_size + max_total_eggs * 2; i++) {
            rand_buffer_host[i] = (float)((double)rand() / RAND_MAX);
        }
        err = clEnqueueWriteBuffer(queue, rand_buffer, CL_TRUE, 0, 
                                  (pop_size + max_total_eggs * 2) * sizeof(float), rand_buffer_host, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing rand_buffer: %d\n", err);
            goto cleanup;
        }

        // Lay eggs
        int total_eggs = 0;
        for (int i = 0; i < pop_size; i++) {
            num_eggs[i] = MIN_EGGS + (int)(rand_buffer_host[i] * (MAX_EGGS - MIN_EGGS + 1));
            total_eggs += num_eggs[i];
        }
        err = clEnqueueWriteBuffer(queue, num_eggs_buffer, CL_TRUE, 0, 
                                  pop_size * sizeof(int), num_eggs, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing num_eggs: %d\n", err);
            goto cleanup;
        }
        err = clSetKernelArg(egg_kernel, 0, sizeof(cl_mem), &cuckoo_positions_buffer);
        err |= clSetKernelArg(egg_kernel, 1, sizeof(cl_mem), &egg_positions_buffer);
        err |= clSetKernelArg(egg_kernel, 2, sizeof(cl_mem), &num_eggs_buffer);
        err |= clSetKernelArg(egg_kernel, 3, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(egg_kernel, 4, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(egg_kernel, 5, sizeof(cl_mem), &rand_buffer);
        err |= clSetKernelArg(egg_kernel, 6, sizeof(int), &dim);
        err |= clSetKernelArg(egg_kernel, 7, sizeof(int), &pop_size);
        err |= clSetKernelArg(egg_kernel, 8, sizeof(int), &total_eggs);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting egg kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, egg_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing egg kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate eggs on CPU
        err = clEnqueueReadBuffer(queue, egg_positions_buffer, CL_TRUE, 0, 
                                 total_eggs * dim * sizeof(float), egg_positions, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading egg_positions: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < total_eggs; i++) {
            for (int d = 0; d < dim; d++) {
                cpu_position[d] = (double)egg_positions[i * dim + d];
            }
            float new_fitness = (float)objective_function(cpu_position);
            if (!isfinite(new_fitness)) {
                fprintf(stderr, "Warning: Non-finite fitness for egg %d: %f\n", i, new_fitness);
                new_fitness = INFINITY;
            }
            all_fitness[pop_size + i] = new_fitness;
        }

        // Combine cuckoos and eggs
        int total_positions = pop_size + total_eggs;
        memcpy(all_positions, cuckoo_positions, pop_size * dim * sizeof(float));
        memcpy(&all_positions[pop_size * dim], egg_positions, total_eggs * dim * sizeof(float));
        err = clEnqueueWriteBuffer(queue, all_positions_buffer, CL_TRUE, 0, 
                                  total_positions * dim * sizeof(float), all_positions, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, all_fitness_buffer, CL_TRUE, 0, 
                                   total_positions * sizeof(float), all_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing all_positions/all_fitness: %d\n", err);
            goto cleanup;
        }

        // Select best cuckoos
        int max_cuckoos = MAX_CUCKOOS;
        err = clSetKernelArg(select_kernel, 0, sizeof(cl_mem), &all_positions_buffer);
        err |= clSetKernelArg(select_kernel, 1, sizeof(cl_mem), &all_fitness_buffer);
        err |= clSetKernelArg(select_kernel, 2, sizeof(cl_mem), &new_positions_buffer);
        err |= clSetKernelArg(select_kernel, 3, sizeof(cl_mem), &new_fitness_buffer);
        err |= clSetKernelArg(select_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(select_kernel, 5, sizeof(int), &total_positions);
        err |= clSetKernelArg(select_kernel, 6, sizeof(int), &max_cuckoos);
        err |= clSetKernelArg(select_kernel, 7, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(select_kernel, 8, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting select kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((total_positions + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, select_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing select kernel: %d\n", err);
            goto cleanup;
        }

        // Update cuckoo population
        err = clEnqueueReadBuffer(queue, new_positions_buffer, CL_TRUE, 0, 
                                 pop_size * dim * sizeof(float), cuckoo_positions, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(queue, new_fitness_buffer, CL_TRUE, 0, 
                                  pop_size * sizeof(float), all_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading new_positions/new_fitness: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueWriteBuffer(queue, cuckoo_positions_buffer, CL_TRUE, 0, 
                                  pop_size * dim * sizeof(float), cuckoo_positions, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing cuckoo_positions: %d\n", err);
            goto cleanup;
        }

        // Compute variance on CPU
        double *means = (double *)calloc(dim, sizeof(double));
        double var_sum = 0.0;
        for (int i = 0; i < pop_size; i++) {
            for (int d = 0; d < dim; d++) {
                means[d] += (double)cuckoo_positions[i * dim + d];
            }
        }
        for (int d = 0; d < dim; d++) {
            means[d] /= pop_size;
        }
        for (int i = 0; i < pop_size; i++) {
            for (int d = 0; d < dim; d++) {
                double diff = (double)cuckoo_positions[i * dim + d] - means[d];
                var_sum += diff * diff;
            }
        }
        var_sum /= pop_size * dim;
        free(means);

        // Check stopping condition
        int stop = (var_sum < VARIANCE_THRESHOLD);

        // Find best cuckoo
        int best_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (all_fitness[i] < all_fitness[best_idx]) {
                best_idx = i;
            }
        }

        // Migrate
        err = clSetKernelArg(migrate_kernel, 0, sizeof(cl_mem), &cuckoo_positions_buffer);
        err |= clSetKernelArg(migrate_kernel, 1, sizeof(cl_mem), &all_fitness_buffer);
        err |= clSetKernelArg(migrate_kernel, 2, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(migrate_kernel, 3, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(migrate_kernel, 4, sizeof(cl_mem), &new_positions_buffer);
        err |= clSetKernelArg(migrate_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(migrate_kernel, 6, sizeof(int), &pop_size);
        err |= clSetKernelArg(migrate_kernel, 7, sizeof(int), &best_idx);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting migrate kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, migrate_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[3]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing migrate kernel: %d\n", err);
            goto cleanup;
        }

        // Update cuckoo positions
        err = clEnqueueReadBuffer(queue, new_positions_buffer, CL_TRUE, 0, 
                                 pop_size * dim * sizeof(float), cuckoo_positions, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading new_positions: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueWriteBuffer(queue, cuckoo_positions_buffer, CL_TRUE, 0, 
                                  pop_size * dim * sizeof(float), cuckoo_positions, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing cuckoo_positions: %d\n", err);
            goto cleanup;
        }

        // Evaluate new population
        for (int i = 0; i < pop_size; i++) {
            for (int d = 0; d < dim; d++) {
                cpu_position[d] = (double)cuckoo_positions[i * dim + d];
            }
            float new_fitness = (float)objective_function(cpu_position);
            if (!isfinite(new_fitness)) {
                fprintf(stderr, "Warning: Non-finite fitness for cuckoo %d: %f\n", i, new_fitness);
                new_fitness = INFINITY;
            }
            all_fitness[i] = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = (double)new_fitness;
                for (int d = 0; d < dim; d++) {
                    opt->best_solution.position[d] = (double)cuckoo_positions[i * dim + d];
                }
            }
        }
        err = clEnqueueWriteBuffer(queue, all_fitness_buffer, CL_TRUE, 0, 
                                  pop_size * sizeof(float), all_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing all_fitness: %d\n", err);
            goto cleanup;
        }

        // Replace worst and second-worst cuckoos
        int worst_idx = 0, second_worst_idx = (pop_size > 1) ? 1 : 0;
        for (int i = 0; i < pop_size; i++) {
            if (all_fitness[i] > all_fitness[worst_idx]) {
                second_worst_idx = worst_idx;
                worst_idx = i;
            } else if (i != worst_idx && all_fitness[i] > all_fitness[second_worst_idx]) {
                second_worst_idx = i;
            }
        }
        if (all_fitness[worst_idx] > opt->best_solution.fitness) {
            for (int d = 0; d < dim; d++) {
                cuckoo_positions[worst_idx * dim + d] = (float)opt->best_solution.position[d];
            }
            all_fitness[worst_idx] = (float)opt->best_solution.fitness;
        }
        if (pop_size > 1) {
            for (int d = 0; d < dim; d++) {
                float new_pos = (float)opt->best_solution.position[d] * rand_buffer_host[d];
                new_pos = fmax(bounds[d * 2], fmin(bounds[d * 2 + 1], new_pos));
                cuckoo_positions[second_worst_idx * dim + d] = new_pos;
            }
            for (int d = 0; d < dim; d++) {
                cpu_position[d] = (double)cuckoo_positions[second_worst_idx * dim + d];
            }
            all_fitness[second_worst_idx] = (float)objective_function(cpu_position);
        }
        err = clEnqueueWriteBuffer(queue, cuckoo_positions_buffer, CL_TRUE, 0, 
                                  pop_size * dim * sizeof(float), cuckoo_positions, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, all_fitness_buffer, CL_TRUE, 0, 
                                   pop_size * sizeof(float), all_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing cuckoo_positions/all_fitness: %d\n", err);
            goto cleanup;
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
        double init_time = 0.0, egg_time = 0.0, select_time = 0.0, migrate_time = 0.0;
        for (int i = 0; i < 4; i++) {
            if (events[i]) {
                err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                err |= clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                if (err == CL_SUCCESS) {
                    double time_ms = (time_end - time_start) / 1e6;
                    if (i == 0) init_time = time_ms;
                    else if (i == 1) egg_time = time_ms;
                    else if (i == 2) select_time = time_ms;
                    else if (i == 3) migrate_time = time_ms;
                }
            }
        }
        end_time = get_time_ms();
        printf("CO|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Egg: %.3f ms | Select: %.3f ms | Migrate: %.3f ms\n",
               iter + 1, opt->best_solution.fitness, end_time - start_time, 
               init_time, egg_time, select_time, migrate_time);

        // Reset events
        for (int i = 0; i < 4; i++) {
            if (events[i]) {
                clReleaseEvent(events[i]);
                events[i] = 0;
            }
        }

        if (stop) {
            break;
        }
    }

    // Update CPU-side population
    err = clEnqueueReadBuffer(queue, cuckoo_positions_buffer, CL_TRUE, 0, 
                             pop_size * dim * sizeof(float), cuckoo_positions, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, all_fitness_buffer, CL_TRUE, 0, 
                              pop_size * sizeof(float), all_fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final buffers: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < pop_size; i++) {
        for (int d = 0; d < dim; d++) {
            float p = cuckoo_positions[i * dim + d];
            float lower_bound = bounds[d * 2];
            float upper_bound = bounds[d * 2 + 1];
            if (p < lower_bound || p > upper_bound || !isfinite(p)) {
                fprintf(stderr, "Warning: Final position out of bounds or non-finite for cuckoo %d, dim %d: %f\n", 
                        i, d, p);
                p = lower_bound + (upper_bound - lower_bound) * ((float)rand() / RAND_MAX);
                cuckoo_positions[i * dim + d] = p;
            }
            opt->population[i].position[d] = (double)p;
        }
        opt->population[i].fitness = (double)all_fitness[i];
    }

cleanup:
    if (cuckoo_positions) free(cuckoo_positions);
    if (egg_positions) free(egg_positions);
    if (all_positions) free(all_positions);
    if (all_fitness) free(all_fitness);
    if (new_positions) free(new_positions);
    if (new_fitness) free(new_fitness);
    if (num_eggs) free(num_eggs);
    if (rng_states) free(rng_states);
    if (bounds) free(bounds);
    if (rand_buffer_host) free(rand_buffer_host);
    if (cpu_position) free(cpu_position);
    if (cuckoo_positions_buffer) clReleaseMemObject(cuckoo_positions_buffer);
    if (egg_positions_buffer) clReleaseMemObject(egg_positions_buffer);
    if (all_positions_buffer) clReleaseMemObject(all_positions_buffer);
    if (all_fitness_buffer) clReleaseMemObject(all_fitness_buffer);
    if (new_positions_buffer) clReleaseMemObject(new_positions_buffer);
    if (new_fitness_buffer) clReleaseMemObject(new_fitness_buffer);
    if (num_eggs_buffer) clReleaseMemObject(num_eggs_buffer);
    if (rng_states_buffer) clReleaseMemObject(rng_states_buffer);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (rand_buffer) clReleaseMemObject(rand_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (egg_kernel) clReleaseKernel(egg_kernel);
    if (select_kernel) clReleaseKernel(select_kernel);
    if (migrate_kernel) clReleaseKernel(migrate_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 4; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) {
        clFinish(queue);
        clReleaseCommandQueue(queue);
    }
}
