#include "CroSA.h"
#include "generaloptimizer.h"
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

// OpenCL kernel source for CroSA
static const char *crosa_kernel_source =
"#define AWARENESS_PROBABILITY 0.1f\n"
"#define FLIGHT_LENGTH 2.0f\n"
// Random number generator
"uint lcg_rand(uint *seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float rand_float(uint *seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
// Initialize kernel
"__kernel void initialize_crosa(\n"
"    __global float *positions,\n"
"    __global float *fitness,\n"
"    __global float *memory_positions,\n"
"    __global float *memory_fitness,\n"
"    __global uint *rng_states,\n"
"    __global const float *bounds,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    __local float *local_bounds)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (idx >= pop_size) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    if (local_id < dim * 2) {\n"
"        local_bounds[local_id] = bounds[local_id];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (idx < pop_size) {\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float lower = local_bounds[d * 2];\n"
"            float upper = local_bounds[d * 2 + 1];\n"
"            float pos = lower + (upper - lower) * rand_float(&rng_state, 0.0f, 1.0f);\n"
"            positions[idx * dim + d] = pos;\n"
"            memory_positions[idx * dim + d] = pos;\n"
"        }\n"
"        fitness[idx] = INFINITY;\n"
"        memory_fitness[idx] = INFINITY;\n"
"    }\n"
"    rng_states[idx] = rng_state;\n"
"}\n"
// Update positions kernel
"__kernel void update_positions(\n"
"    __global float *positions,\n"
"    __global float *memory_positions,\n"
"    __global uint *rng_states,\n"
"    __global const float *bounds,\n"
"    __global float *new_positions,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    __local float *local_bounds)\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (idx >= pop_size) return;\n"
"    uint rng_state = rng_states[idx];\n"
"    if (local_id < dim * 2) {\n"
"        local_bounds[local_id] = bounds[local_id];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    float r = rand_float(&rng_state, 0.0f, 1.0f);\n"
"    int random_crow = (int)(rand_float(&rng_state, 0.0f, (float)pop_size));\n"
"    if (random_crow == idx) random_crow = (random_crow + 1) % pop_size;\n"
"    if (r > AWARENESS_PROBABILITY) {\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float rand_factor = FLIGHT_LENGTH * rand_float(&rng_state, 0.0f, 1.0f);\n"
"            float new_pos = positions[idx * dim + d] + rand_factor * (memory_positions[random_crow * dim + d] - positions[idx * dim + d]);\n"
"            float lower = local_bounds[d * 2];\n"
"            float upper = local_bounds[d * 2 + 1];\n"
"            new_pos = fmin(fmax(new_pos, lower), upper);\n"
"            new_positions[idx * dim + d] = new_pos;\n"
"        }\n"
"    } else {\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float lower = local_bounds[d * 2];\n"
"            float upper = local_bounds[d * 2 + 1];\n"
"            float new_pos = lower + (upper - lower) * rand_float(&rng_state, 0.0f, 1.0f);\n"
"            new_positions[idx * dim + d] = new_pos;\n"
"        }\n"
"    }\n"
"    rng_states[idx] = rng_state;\n"
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
void CroSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, update_kernel = NULL, best_kernel = NULL;
    cl_command_queue queue = NULL;
    cl_mem positions_buffer = NULL, fitness_buffer = NULL;
    cl_mem memory_positions_buffer = NULL, memory_fitness_buffer = NULL;
    cl_mem rng_states_buffer = NULL, bounds_buffer = NULL;
    cl_mem new_positions_buffer = NULL, best_idx_buffer = NULL;
    float *positions = NULL, *fitness = NULL, *memory_positions = NULL, *memory_fitness = NULL;
    float *bounds = NULL, *new_positions = NULL;
    uint *rng_states = NULL;
    double *cpu_position = NULL;
    cl_event events[3] = {0};
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

    // Initialize OpenCL program
    program = clCreateProgramWithSource(opt->context, 1, &crosa_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating CroSA program: %d\n", err);
        goto cleanup;
    }
    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building CroSA program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        goto cleanup;
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_crosa", &err);
    update_kernel = clCreateKernel(program, "update_positions", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating CroSA kernels: %d\n", err);
        goto cleanup;
    }

    // Create command queue with profiling
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
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
    size_t local_work_size = max_work_group_size < 64 ? max_work_group_size : 64;

    // Allocate GPU buffers
    positions_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(float), NULL, &err);
    memory_positions_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    memory_fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(float), NULL, &err);
    rng_states_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(uint), NULL, &err);
    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, dim * 2 * sizeof(float), NULL, &err);
    new_positions_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, pop_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating CroSA buffers: %d\n", err);
        goto cleanup;
    }

    // Allocate host memory
    positions = (float *)malloc(pop_size * dim * sizeof(float));
    fitness = (float *)malloc(pop_size * sizeof(float));
    memory_positions = (float *)malloc(pop_size * dim * sizeof(float));
    memory_fitness = (float *)malloc(pop_size * sizeof(float));
    bounds = (float *)malloc(dim * 2 * sizeof(float));
    new_positions = (float *)malloc(pop_size * dim * sizeof(float));
    rng_states = (uint *)malloc(pop_size * sizeof(uint));
    cpu_position = (double *)malloc(dim * sizeof(double));
    if (!positions || !fitness || !memory_positions || !memory_fitness || 
        !bounds || !new_positions || !rng_states || !cpu_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }

    // Initialize host data
    for (int d = 0; d < dim; d++) {
        bounds[d * 2] = (float)opt->bounds[d * 2];
        bounds[d * 2 + 1] = (float)opt->bounds[d * 2 + 1];
    }
    for (int i = 0; i < pop_size; i++) {
        rng_states[i] = (uint)(time(NULL) ^ (i + 1));
        for (int d = 0; d < dim; d++) {
            positions[i * dim + d] = (float)opt->population[i].position[d];
            memory_positions[i * dim + d] = (float)opt->population[i].position[d];
        }
        fitness[i] = INFINITY;
        memory_fitness[i] = INFINITY;
    }

    // Write initial data to GPU
    err = clEnqueueWriteBuffer(queue, positions_buffer, CL_TRUE, 0, 
                              pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(float), fitness, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, memory_positions_buffer, CL_TRUE, 0, 
                               pop_size * dim * sizeof(float), memory_positions, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, memory_fitness_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(float), memory_fitness, 0, NULL, NULL);
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
    err |= clSetKernelArg(init_kernel, 2, sizeof(cl_mem), &memory_positions_buffer);
    err |= clSetKernelArg(init_kernel, 3, sizeof(cl_mem), &memory_fitness_buffer);
    err |= clSetKernelArg(init_kernel, 4, sizeof(cl_mem), &rng_states_buffer);
    err |= clSetKernelArg(init_kernel, 5, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 6, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 7, sizeof(int), &pop_size);
    err |= clSetKernelArg(init_kernel, 8, 2 * dim * sizeof(float), NULL);
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
            memory_positions[i * dim + d] = positions[i * dim + d];
        }
        float new_fitness = (float)objective_function(cpu_position);
        if (!isfinite(new_fitness)) {
            fprintf(stderr, "Warning: Non-finite fitness for crow %d: %f\n", i, new_fitness);
            new_fitness = INFINITY;
        }
        fitness[i] = new_fitness;
        memory_fitness[i] = new_fitness;
    }
    err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                              pop_size * sizeof(float), fitness, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, memory_positions_buffer, CL_TRUE, 0, 
                               pop_size * dim * sizeof(float), memory_positions, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, memory_fitness_buffer, CL_TRUE, 0, 
                               pop_size * sizeof(float), memory_fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial fitness/memory: %d\n", err);
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

        // Update positions
        err = clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &positions_buffer);
        err |= clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &memory_positions_buffer);
        err |= clSetKernelArg(update_kernel, 2, sizeof(cl_mem), &rng_states_buffer);
        err |= clSetKernelArg(update_kernel, 3, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(update_kernel, 4, sizeof(cl_mem), &new_positions_buffer);
        err |= clSetKernelArg(update_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(update_kernel, 6, sizeof(int), &pop_size);
        err |= clSetKernelArg(update_kernel, 7, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting update kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, update_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing update kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate new positions on CPU
        err = clEnqueueReadBuffer(queue, new_positions_buffer, CL_TRUE, 0, 
                                 pop_size * dim * sizeof(float), new_positions, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading new_positions: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < pop_size; i++) {
            for (int d = 0; d < dim; d++) {
                cpu_position[d] = (double)new_positions[i * dim + d];
            }
            float new_fitness = (float)objective_function(cpu_position);
            if (!isfinite(new_fitness)) {
                fprintf(stderr, "Warning: Non-finite fitness for crow %d: %f\n", i, new_fitness);
                new_fitness = INFINITY;
            }
            // Update memory and positions
            if (new_fitness < memory_fitness[i]) {
                memory_fitness[i] = new_fitness;
                for (int d = 0; d < dim; d++) {
                    memory_positions[i * dim + d] = new_positions[i * dim + d];
                    positions[i * dim + d] = new_positions[i * dim + d];
                }
                fitness[i] = new_fitness;
            } else {
                for (int d = 0; d < dim; d++) {
                    positions[i * dim + d] = new_positions[i * dim + d];
                }
                fitness[i] = new_fitness;
            }
        }

        // Write updated data to GPU
        err = clEnqueueWriteBuffer(queue, positions_buffer, CL_TRUE, 0, 
                                  pop_size * dim * sizeof(float), positions, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, 
                                   pop_size * sizeof(float), fitness, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, memory_positions_buffer, CL_TRUE, 0, 
                                   pop_size * dim * sizeof(float), memory_positions, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, memory_fitness_buffer, CL_TRUE, 0, 
                                   pop_size * sizeof(float), memory_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing positions/fitness/memory: %d\n", err);
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
        err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
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

        // Profiling
        err = clFinish(queue);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error finishing queue: %d\n", err);
            goto cleanup;
        }
        cl_ulong time_start, time_end;
        double init_time = 0.0, update_time = 0.0, best_time = 0.0;
        for (int i = 0; i < 3; i++) {
            if (events[i]) {
                err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                err |= clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                if (err == CL_SUCCESS) {
                    double time_ms = (time_end - time_start) / 1e6;
                    if (i == 0) init_time = time_ms;
                    else if (i == 1) update_time = time_ms;
                    else if (i == 2) best_time = time_ms;
                }
            }
        }
        end_time = get_time_ms();
        printf("CroSA|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Update: %.3f ms | Best: %.3f ms\n",
               iter + 1, opt->best_solution.fitness, end_time - start_time, 
               init_time, update_time, best_time);

        // Reset events
        for (int i = 0; i < 3; i++) {
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
                fprintf(stderr, "Warning: Final position out of bounds or non-finite for crow %d, dim %d: %f\n", i, d, p);
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
    if (memory_positions) free(memory_positions);
    if (memory_fitness) free(memory_fitness);
    if (bounds) free(bounds);
    if (new_positions) free(new_positions);
    if (rng_states) free(rng_states);
    if (cpu_position) free(cpu_position);
    if (positions_buffer) clReleaseMemObject(positions_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (memory_positions_buffer) clReleaseMemObject(memory_positions_buffer);
    if (memory_fitness_buffer) clReleaseMemObject(memory_fitness_buffer);
    if (rng_states_buffer) clReleaseMemObject(rng_states_buffer);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (new_positions_buffer) clReleaseMemObject(new_positions_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (update_kernel) clReleaseKernel(update_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 3; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) {
        clFinish(queue);
        clReleaseCommandQueue(queue);
    }
}
