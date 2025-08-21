/* ILS.c - GPU-Optimized Iterated Local Search with Flexible Objective Function */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
#include "generaloptimizer.h"
#include "ILS.h"

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
static const char* ils_kernel_source =
"// Simple LCG random number generator\n"
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
"// Check if a point is within bounds\n"
"bool in_bounds(__global const float* point, __local const float* bounds, const int dim) {\n"
"    for (int i = 0; i < dim; i++) {\n"
"        if (point[i] < bounds[2 * i] || point[i] > bounds[2 * i + 1]) {\n"
"            return false;\n"
"        }\n"
"    }\n"
"    return true;\n"
"}\n"
"__kernel void initialize_solutions(\n"
"    __global float* solutions,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int num_solutions,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < num_solutions) {\n"
"        uint local_seed = seed + id * dim;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            solutions[id * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void hill_climb_step(\n"
"    __global float* solutions,\n"
"    __global float* fitness,\n"
"    __global float* candidates,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int num_solutions,\n"
"    const float step_size,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < num_solutions) {\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        uint local_seed = seed + id * dim;\n"
"        bool valid = false;\n"
"        int attempts = 0;\n"
"        const int max_attempts = 10;\n"
"        while (!valid && attempts < max_attempts) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float delta = lcg_rand_float(&local_seed, -step_size, step_size);\n"
"                candidates[id * dim + d] = solutions[id * dim + d] + delta;\n"
"            }\n"
"            valid = in_bounds(&candidates[id * dim], local_bounds, dim);\n"
"            attempts++;\n"
"        }\n"
"        if (!valid) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                candidates[id * dim + d] = solutions[id * dim + d];\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void perturb_solutions(\n"
"    __global float* solutions,\n"
"    __global const float* best_solution,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int num_solutions,\n"
"    const float perturbation_size,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < num_solutions) {\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        uint local_seed = seed + id * dim;\n"
"        bool valid = false;\n"
"        int attempts = 0;\n"
"        const int max_attempts = 10;\n"
"        while (!valid && attempts < max_attempts) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float delta = lcg_rand_float(&local_seed, -perturbation_size, perturbation_size);\n"
"                solutions[id * dim + d] = best_solution[d] + delta;\n"
"            }\n"
"            valid = in_bounds(&solutions[id * dim], local_bounds, dim);\n"
"            attempts++;\n"
"        }\n"
"        if (!valid) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                solutions[id * dim + d] = best_solution[d];\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void update_solutions(\n"
"    __global float* solutions,\n"
"    __global float* fitness,\n"
"    __global const float* candidates,\n"
"    __global const float* candidate_fitness,\n"
"    const int dim,\n"
"    const int num_solutions)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < num_solutions) {\n"
"        if (candidate_fitness[id] < fitness[id]) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                solutions[id * dim + d] = candidates[id * dim + d];\n"
"            }\n"
"            fitness[id] = candidate_fitness[id];\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void find_best(\n"
"    __global const float* fitness,\n"
"    __global int* best_idx,\n"
"    const int num_solutions,\n"
"    __local float* local_fitness,\n"
"    __local int* local_indices)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    if (id < num_solutions) {\n"
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

void ILS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, hill_climb_kernel = NULL, perturb_kernel = NULL, update_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, solutions_buffer = NULL, fitness_buffer = NULL, candidates_buffer = NULL, candidate_fitness_buffer = NULL, best_solution_buffer = NULL, best_idx_buffer = NULL;
    float *bounds_float = NULL, *solutions = NULL, *fitness = NULL, *candidates = NULL, *candidate_fitness = NULL, *best_solution = NULL;
    int *best_idx_array = NULL;
    cl_event events[5] = {0};
    double start_time, end_time;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    double *temp_position = NULL;

    // Validate input
    if (!opt || !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        return;
    }
    const int dim = opt->dim;
    const int n_restarts = 30;
    const int n_iterations = 1000;
    const int num_solutions = 256; // Number of parallel solutions per restart
    if (dim < 1 || dim > 32) {
        fprintf(stderr, "Error: Invalid dim (%d), must be between 1 and 32\n", dim);
        return;
    }

    // Select GPU platform and device
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "Error: No OpenCL platforms found: %d\n", err);
        return;
    }
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) {
        fprintf(stderr, "Error: Memory allocation failed for platforms\n");
        return;
    }
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error getting platform IDs: %d\n", err);
        free(platforms);
        return;
    }

    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) continue;
        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        if (!devices) continue;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
        if (err != CL_SUCCESS) {
            free(devices);
            continue;
        }
        for (cl_uint j = 0; j < num_devices; j++) {
            char device_name[128] = "Unknown";
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            if (strstr(device_name, "Intel(R) Graphics [0x46a8]")) {
                platform = platforms[i];
                device = devices[j];
            }
        }
        free(devices);
        if (platform && device) break;
    }
    free(platforms);
    if (!platform || !device) {
        fprintf(stderr, "Error: No suitable GPU device found\n");
        return;
    }

    // Create context and queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !context) {
        fprintf(stderr, "Error creating OpenCL context: %d\n", err);
        return;
    }
    cl_queue_properties queue_props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, queue_props, &err);
    if (err != CL_SUCCESS || !queue) {
        fprintf(stderr, "Error creating OpenCL queue: %d\n", err);
        clReleaseContext(context);
        return;
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
    program = clCreateProgramWithSource(context, 1, &ils_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating ILS program: %d\n", err);
        goto cleanup;
    }
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building ILS program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        program = NULL;
        goto cleanup;
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_solutions", &err);
    hill_climb_kernel = clCreateKernel(program, "hill_climb_step", &err);
    perturb_kernel = clCreateKernel(program, "perturb_solutions", &err);
    update_kernel = clCreateKernel(program, "update_solutions", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !hill_climb_kernel || !perturb_kernel || !update_kernel || !best_kernel) {
        fprintf(stderr, "Error creating ILS kernels: %d\n", err);
        goto cleanup;
    }

    // Allocate host memory
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    solutions = (float*)malloc(num_solutions * dim * sizeof(float));
    fitness = (float*)malloc(num_solutions * sizeof(float));
    candidates = (float*)malloc(num_solutions * dim * sizeof(float));
    candidate_fitness = (float*)malloc(num_solutions * sizeof(float));
    best_solution = (float*)malloc(dim * sizeof(float));
    best_idx_array = (int*)malloc(num_solutions * sizeof(int));
    temp_position = (double*)malloc(dim * sizeof(double));
    if (!bounds_float || !solutions || !fitness || !candidates || !candidate_fitness || !best_solution || !best_idx_array || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }
    for (int i = 0; i < dim; i++) {
        best_solution[i] = 0.0f;
    }

    // Create buffers
    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    solutions_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, num_solutions * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, num_solutions * sizeof(float), NULL, &err);
    candidates_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, num_solutions * dim * sizeof(float), NULL, &err);
    candidate_fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, num_solutions * sizeof(float), NULL, &err);
    best_solution_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, num_solutions * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !solutions_buffer || !fitness_buffer || !candidates_buffer || 
        !candidate_fitness_buffer || !best_solution_buffer || !best_idx_buffer) {
        fprintf(stderr, "Error creating ILS buffers: %d\n", err);
        goto cleanup;
    }

    // Write bounds and initial best solution
    err = clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, best_solution_buffer, CL_TRUE, 0, dim * sizeof(float), best_solution, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial buffers: %d\n", err);
        goto cleanup;
    }

    uint seed = (uint)time(NULL);
    float best_fitness = INFINITY;

    // Main ILS loop
    for (int restart = 0; restart < n_restarts; restart++) {
        start_time = get_time_ms();

        // Initialize solutions
        err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &solutions_buffer);
        err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(init_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(init_kernel, 3, sizeof(int), &num_solutions);
        err |= clSetKernelArg(init_kernel, 4, sizeof(uint), &seed);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting init kernel args: %d\n", err);
            goto cleanup;
        }
        size_t global_work_size = ((num_solutions + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[0]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing init kernel: %d\n", err);
            goto cleanup;
        }

        // Evaluate initial solutions
        err = clEnqueueReadBuffer(queue, solutions_buffer, CL_TRUE, 0, num_solutions * dim * sizeof(float), solutions, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading solutions buffer: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < num_solutions; i++) {
            for (int d = 0; d < dim; d++) {
                temp_position[d] = (double)solutions[i * dim + d];
            }
            fitness[i] = (float)objective_function(temp_position);
        }
        err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, num_solutions * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            goto cleanup;
        }

        // Hill climbing iterations
        for (int iter = 0; iter < n_iterations; iter++) {
            // Generate candidates
            float step_size = ILS_STEP_SIZE;
            err = clSetKernelArg(hill_climb_kernel, 0, sizeof(cl_mem), &solutions_buffer);
            err |= clSetKernelArg(hill_climb_kernel, 1, sizeof(cl_mem), &fitness_buffer);
            err |= clSetKernelArg(hill_climb_kernel, 2, sizeof(cl_mem), &candidates_buffer);
            err |= clSetKernelArg(hill_climb_kernel, 3, sizeof(cl_mem), &bounds_buffer);
            err |= clSetKernelArg(hill_climb_kernel, 4, sizeof(int), &dim);
            err |= clSetKernelArg(hill_climb_kernel, 5, sizeof(int), &num_solutions);
            err |= clSetKernelArg(hill_climb_kernel, 6, sizeof(float), &step_size);
            err |= clSetKernelArg(hill_climb_kernel, 7, sizeof(uint), &seed);
            err |= clSetKernelArg(hill_climb_kernel, 8, 2 * dim * sizeof(float), NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error setting hill climb kernel args: %d\n", err);
                goto cleanup;
            }
            err = clEnqueueNDRangeKernel(queue, hill_climb_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error enqueuing hill climb kernel: %d\n", err);
                goto cleanup;
            }

            // Evaluate candidates
            err = clEnqueueReadBuffer(queue, candidates_buffer, CL_TRUE, 0, num_solutions * dim * sizeof(float), candidates, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading candidates buffer: %d\n", err);
                goto cleanup;
            }
            for (int i = 0; i < num_solutions; i++) {
                for (int d = 0; d < dim; d++) {
                    temp_position[d] = (double)candidates[i * dim + d];
                }
                candidate_fitness[i] = (float)objective_function(temp_position);
            }
            err = clEnqueueWriteBuffer(queue, candidate_fitness_buffer, CL_TRUE, 0, num_solutions * sizeof(float), candidate_fitness, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing candidate fitness buffer: %d\n", err);
                goto cleanup;
            }

            // Update solutions
            err = clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &solutions_buffer);
            err |= clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &fitness_buffer);
            err |= clSetKernelArg(update_kernel, 2, sizeof(cl_mem), &candidates_buffer);
            err |= clSetKernelArg(update_kernel, 3, sizeof(cl_mem), &candidate_fitness_buffer);
            err |= clSetKernelArg(update_kernel, 4, sizeof(int), &dim);
            err |= clSetKernelArg(update_kernel, 5, sizeof(int), &num_solutions);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error setting update kernel args: %d\n", err);
                goto cleanup;
            }
            err = clEnqueueNDRangeKernel(queue, update_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error enqueuing update kernel: %d\n", err);
                goto cleanup;
            }
        }

        // Find best solution
        err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &best_idx_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(int), &num_solutions);
        err |= clSetKernelArg(best_kernel, 3, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 4, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[3]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing best kernel: %d\n", err);
            goto cleanup;
        }

        // Update global best
        int best_idx;
        err = clEnqueueReadBuffer(queue, best_idx_buffer, CL_TRUE, 0, sizeof(int), &best_idx, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best_idx buffer: %d\n", err);
            goto cleanup;
        }
        float current_fitness;
        err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, best_idx * sizeof(float), sizeof(float), &current_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading current fitness: %d\n", err);
            goto cleanup;
        }
        if (current_fitness < best_fitness) {
            best_fitness = current_fitness;
            err = clEnqueueReadBuffer(queue, solutions_buffer, CL_TRUE, best_idx * dim * sizeof(float), dim * sizeof(float), best_solution, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best solution: %d\n", err);
                goto cleanup;
            }
            err = clEnqueueWriteBuffer(queue, best_solution_buffer, CL_TRUE, 0, dim * sizeof(float), best_solution, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing best solution buffer: %d\n", err);
                goto cleanup;
            }
            opt->best_solution.fitness = (double)best_fitness;
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)best_solution[d];
            }
        }

        // Perturb solutions
        float perturbation_size = ILS_PERTURBATION_SIZE;
        err = clSetKernelArg(perturb_kernel, 0, sizeof(cl_mem), &solutions_buffer);
        err |= clSetKernelArg(perturb_kernel, 1, sizeof(cl_mem), &best_solution_buffer);
        err |= clSetKernelArg(perturb_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(perturb_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(perturb_kernel, 4, sizeof(int), &num_solutions);
        err |= clSetKernelArg(perturb_kernel, 5, sizeof(float), &perturbation_size);
        err |= clSetKernelArg(perturb_kernel, 6, sizeof(uint), &seed);
        err |= clSetKernelArg(perturb_kernel, 7, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting perturb kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, perturb_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[4]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing perturb kernel: %d\n", err);
            goto cleanup;
        }

        // Profiling
        cl_ulong time_start, time_end;
        double init_time = 0, hill_climb_time = 0, update_time = 0, best_time = 0, perturb_time = 0;
        cl_ulong queue_properties;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(cl_ulong), &queue_properties, NULL);
        if (err == CL_SUCCESS && (queue_properties & CL_QUEUE_PROFILING_ENABLE)) {
            for (int i = 0; i < 5; i++) {
                if (events[i]) {
                    err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                    if (err == CL_SUCCESS) {
                        err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                        if (err == CL_SUCCESS) {
                            double time_ms = (time_end - time_start) / 1e6;
                            if (i == 0) init_time = time_ms;
                            else if (i == 1) hill_climb_time = time_ms;
                            else if (i == 2) update_time = time_ms;
                            else if (i == 3) best_time = time_ms;
                            else if (i == 4) perturb_time = time_ms;
                        }
                    }
                }
            }
            end_time = get_time_ms();
            printf("ILS|Restart %2d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | HillClimb: %.3f ms | Update: %.3f ms | Best: %.3f ms | Perturb: %.3f ms\n",
                   restart + 1, opt->best_solution.fitness, end_time - start_time, init_time, hill_climb_time, update_time, best_time, perturb_time);
        } else {
            end_time = get_time_ms();
            printf("ILS|Restart %2d -----> %9.16f | Total: %.3f ms | Profiling disabled\n",
                   restart + 1, opt->best_solution.fitness, end_time - start_time);
        }
    }

cleanup:
    if (bounds_float) free(bounds_float);
    if (solutions) free(solutions);
    if (fitness) free(fitness);
    if (candidates) free(candidates);
    if (candidate_fitness) free(candidate_fitness);
    if (best_solution) free(best_solution);
    if (best_idx_array) free(best_idx_array);
    if (temp_position) free(temp_position);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (solutions_buffer) clReleaseMemObject(solutions_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (candidates_buffer) clReleaseMemObject(candidates_buffer);
    if (candidate_fitness_buffer) clReleaseMemObject(candidate_fitness_buffer);
    if (best_solution_buffer) clReleaseMemObject(best_solution_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (hill_climb_kernel) clReleaseKernel(hill_climb_kernel);
    if (perturb_kernel) clReleaseKernel(perturb_kernel);
    if (update_kernel) clReleaseKernel(update_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 5; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (queue) clFinish(queue);
}
