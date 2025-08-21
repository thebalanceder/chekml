#include "LS.h"
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

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
static const char* ls_kernel_source =
"#define STEP_SIZE 1.0f\n"
"#define NEIGHBOR_COUNT 1024\n"
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
"__kernel void initialize_solution(\n"
"    __global float* solution,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < dim) {\n"
"        uint local_seed = seed + id;\n"
"        float min = bounds[2 * id];\n"
"        float max = bounds[2 * id + 1];\n"
"        solution[id] = min + (max - min) * lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"    }\n"
"}\n"
"__kernel void generate_neighbors(\n"
"    __global const float* current_solution,\n"
"    __global float* neighbors,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int neighbor_idx = id / dim;\n"
"    int dim_idx = id % dim;\n"
"    if (neighbor_idx < NEIGHBOR_COUNT && dim_idx < dim) {\n"
"        uint local_seed = seed + id;\n"
"        float perturbation = lcg_rand_float(&local_seed, -STEP_SIZE, STEP_SIZE);\n"
"        float pos = current_solution[dim_idx] + perturbation;\n"
"        float min = bounds[2 * dim_idx];\n"
"        float max = bounds[2 * dim_idx + 1];\n"
"        neighbors[neighbor_idx * dim + dim_idx] = pos < min ? min : (pos > max ? max : pos);\n"
"    }\n"
"}\n"
"__kernel void find_best_neighbor(\n"
"    __global const float* fitness,\n"
"    __global int* best_idx,\n"
"    __global float* best_fitness,\n"
"    const int neighbor_count,\n"
"    __local float* local_fitness,\n"
"    __local int* local_indices)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    if (id < neighbor_count) {\n"
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
"        best_fitness[get_group_id(0)] = local_fitness[0];\n"
"    }\n"
"}\n";

void LS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, neighbor_kernel = NULL, reduce_kernel = NULL;
    cl_mem solution_buffer = NULL, neighbors_buffer = NULL, fitness_buffer = NULL;
    cl_mem bounds_buffer = NULL, best_idx_buffer = NULL, best_fitness_buffer = NULL;
    float *current_solution = NULL, *neighbors = NULL, *fitness = NULL, *bounds = NULL;
    double *temp_solution = NULL;
    cl_event events[3] = {0};
    double start_time, end_time;

    // Validate input
    if (!opt || !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    const int dim = opt->dim;
    const int neighbor_count = NEIGHBOR_COUNT; // 1024 from LS.h
    const int max_iter = MAX_ITER;

    if (dim < 1 || neighbor_count < 1 || max_iter < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), neighbor_count (%d), or max_iter (%d)\n", 
                dim, neighbor_count, max_iter);
        exit(1);
    }

    // Select GPU platform and device
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "Error: No OpenCL platforms found: %d\n", err);
        exit(1);
    }
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) {
        fprintf(stderr, "Error: Memory allocation failed for platforms\n");
        exit(1);
    }
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error getting platform IDs: %d\n", err);
        free(platforms);
        exit(1);
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
            fprintf(stderr, "Error: Memory allocation failed for devices\n");
            free(platforms);
            exit(1);
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
        exit(1);
    }

    // Create context and queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !context) {
        fprintf(stderr, "Error creating OpenCL context: %d\n", err);
        exit(1);
    }
    cl_queue_properties queue_props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, queue_props, &err);
    if (err != CL_SUCCESS || !queue) {
        fprintf(stderr, "Error creating OpenCL queue: %d\n", err);
        clReleaseContext(context);
        exit(1);
    }

    // Query device capabilities
    size_t max_work_group_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying CL_DEVICE_MAX_WORK_GROUP_SIZE: %d\n", err);
        goto cleanup;
    }
    size_t local_work_size = max_work_group_size < 64 ? max_work_group_size : 64;

    // Allocate host memory
    current_solution = (float*)malloc(dim * sizeof(float));
    neighbors = (float*)malloc(neighbor_count * dim * sizeof(float));
    fitness = (float*)malloc(neighbor_count * sizeof(float));
    bounds = (float*)malloc(2 * dim * sizeof(float));
    temp_solution = (double*)malloc(dim * sizeof(double));
    if (!current_solution || !neighbors || !fitness || !bounds || !temp_solution) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds[i] = (float)opt->bounds[i];
    }

    // Create buffers
    solution_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    neighbors_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, neighbor_count * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, neighbor_count * sizeof(float), NULL, &err);
    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, neighbor_count * sizeof(int), NULL, &err);
    best_fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, neighbor_count * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !solution_buffer || !neighbors_buffer || !fitness_buffer ||
        !bounds_buffer || !best_idx_buffer || !best_fitness_buffer) {
        fprintf(stderr, "Error creating buffers: %d\n", err);
        goto cleanup;
    }

    // Write bounds
    err = clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bounds buffer: %d\n", err);
        goto cleanup;
    }

    // Create program
    program = clCreateProgramWithSource(context, 1, &ls_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating program: %d\n", err);
        goto cleanup;
    }
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        goto cleanup;
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_solution", &err);
    neighbor_kernel = clCreateKernel(program, "generate_neighbors", &err);
    reduce_kernel = clCreateKernel(program, "find_best_neighbor", &err);
    if (err != CL_SUCCESS || !init_kernel || !neighbor_kernel || !reduce_kernel) {
        fprintf(stderr, "Error creating kernels: %d\n", err);
        goto cleanup;
    }

    // Initialize solution
    uint seed = (uint)time(NULL);
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &solution_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 3, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting init kernel args: %d\n", err);
        goto cleanup;
    }
    size_t init_global_work_size = ((dim + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, &init_global_work_size, &local_work_size, 0, NULL, &events[0]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel: %d\n", err);
        goto cleanup;
    }

    // Evaluate initial solution on CPU
    err = clEnqueueReadBuffer(queue, solution_buffer, CL_TRUE, 0, dim * sizeof(float), current_solution, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading initial solution: %d\n", err);
        goto cleanup;
    }
    printf("Initial solution: [");
    for (int i = 0; i < dim; i++) {
        temp_solution[i] = (double)current_solution[i];
        printf("%f", current_solution[i]);
        if (i < dim - 1) printf(", ");
    }
    printf("]\n");
    double current_fitness = objective_function(temp_solution);
    printf("Initial fitness: %f\n", current_fitness);
    opt->best_solution.fitness = current_fitness;
    for (int j = 0; j < dim; j++) {
        opt->best_solution.position[j] = (double)current_solution[j];
    }

    // Optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Generate neighbors on GPU
        err = clSetKernelArg(neighbor_kernel, 0, sizeof(cl_mem), &solution_buffer);
        err |= clSetKernelArg(neighbor_kernel, 1, sizeof(cl_mem), &neighbors_buffer);
        err |= clSetKernelArg(neighbor_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(neighbor_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(neighbor_kernel, 4, sizeof(uint), &seed);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting neighbor kernel args: %d\n", err);
            goto cleanup;
        }
        size_t neighbor_global_work_size = ((neighbor_count * dim + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, neighbor_kernel, 1, NULL, &neighbor_global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing neighbor kernel: %d\n", err);
            goto cleanup;
        }

        // Read neighbors to host
        err = clEnqueueReadBuffer(queue, neighbors_buffer, CL_TRUE, 0, neighbor_count * dim * sizeof(float), neighbors, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading neighbors: %d\n", err);
            goto cleanup;
        }

        // Evaluate neighbors on CPU
        printf("Iteration %d: Evaluating %d neighbors, Current fitness: %f\n", iter + 1, neighbor_count, current_fitness);
        for (int i = 0; i < neighbor_count; i++) {
            for (int j = 0; j < dim; j++) {
                temp_solution[j] = (double)neighbors[i * dim + j];
            }
            fitness[i] = (float)objective_function(temp_solution);
            if (i < 5) { // Print first few for debugging
                printf("  Neighbor %d: [", i);
                for (int j = 0; j < dim; j++) {
                    printf("%f", neighbors[i * dim + j]);
                    if (j < dim - 1) printf(", ");
                }
                printf("], Fitness: %f\n", fitness[i]);
            }
        }

        // Write fitness to GPU
        err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, neighbor_count * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            goto cleanup;
        }

        // Find best neighbor on GPU
        err = clSetKernelArg(reduce_kernel, 0, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(reduce_kernel, 1, sizeof(cl_mem), &best_idx_buffer);
        err |= clSetKernelArg(reduce_kernel, 2, sizeof(cl_mem), &best_fitness_buffer);
        err |= clSetKernelArg(reduce_kernel, 3, sizeof(int), &neighbor_count);
        err |= clSetKernelArg(reduce_kernel, 4, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(reduce_kernel, 5, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting reduce kernel args: %d\n", err);
            goto cleanup;
        }
        size_t reduce_global_work_size = ((neighbor_count + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, reduce_kernel, 1, NULL, &reduce_global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing reduce kernel: %d\n", err);
            goto cleanup;
        }

        // Read best fitness and index
        int best_idx = -1;
        float best_fitness = INFINITY;
        err = clEnqueueReadBuffer(queue, best_idx_buffer, CL_TRUE, 0, sizeof(int), &best_idx, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best_idx: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueReadBuffer(queue, best_fitness_buffer, CL_TRUE, 0, sizeof(float), &best_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best_fitness: %d\n", err);
            goto cleanup;
        }
        printf("Best neighbor idx: %d, Fitness: %f\n", best_idx, best_fitness);

        // Validate best_idx
        if (best_idx < 0 || best_idx >= neighbor_count) {
            printf("Warning: Invalid best_idx (%d), continuing to next iteration\n", best_idx);
            continue;
        }

        // Stop if no better neighbor found
        if (best_fitness >= current_fitness) {
            printf("Stopping at iteration %d: No better neighbor found.\n", iter + 1);
            break;
        }

        // Update current solution
        for (int j = 0; j < dim; j++) {
            current_solution[j] = neighbors[best_idx * dim + j];
        }
        err = clEnqueueWriteBuffer(queue, solution_buffer, CL_TRUE, 0, dim * sizeof(float), current_solution, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing current solution: %d\n", err);
            goto cleanup;
        }
        current_fitness = best_fitness;

        // Update global best
        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)best_fitness;
            for (int j = 0; j < dim; j++) {
                opt->best_solution.position[j] = (double)current_solution[j];
            }
            printf("Iteration %d: Best Value = %f, Solution = [", iter + 1, opt->best_solution.fitness);
            for (int j = 0; j < dim; j++) {
                printf("%f", current_solution[j]);
                if (j < dim - 1) printf(", ");
            }
            printf("]\n");
        }

        // Profiling
        cl_ulong time_start, time_end;
        cl_ulong queue_props;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(cl_ulong), &queue_props, NULL);
        if (err == CL_SUCCESS && (queue_props & CL_QUEUE_PROFILING_ENABLE)) {
            for (int i = 0; i < 3; i++) {
                if (events[i]) {
                    err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                    if (err == CL_SUCCESS) {
                        err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                        if (err == CL_SUCCESS) {
                            double time_ms = (time_end - time_start) / 1e6;
                            printf("Event %d: %.3f ms\n", i, time_ms);
                            if (i == 0) printf("  Init: %.3f ms\n", time_ms);
                            else if (i == 1) printf("  Neighbor: %.3f ms\n", time_ms);
                            else if (i == 2) printf("  Reduce: %.3f ms\n", time_ms);
                        }
                    }
                    clReleaseEvent(events[i]);
                    events[i] = NULL;
                }
            }
            end_time = get_time_ms();
            printf("LS|%d: Total: %.3f ms\n", iter + 1, end_time - start_time);
        }
    }

    // Ensure bounds are respected
    enforce_bound_constraints(opt);

cleanup:
    clFinish(queue); // Ensure all operations complete
    if (current_solution) free(current_solution);
    if (neighbors) free(neighbors);
    if (fitness) free(fitness);
    if (bounds) free(bounds);
    if (temp_solution) free(temp_solution);
    if (solution_buffer) clReleaseMemObject(solution_buffer);
    if (neighbors_buffer) clReleaseMemObject(neighbors_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (best_fitness_buffer) clReleaseMemObject(best_fitness_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (neighbor_kernel) clReleaseKernel(neighbor_kernel);
    if (reduce_kernel) clReleaseKernel(reduce_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 3; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}
