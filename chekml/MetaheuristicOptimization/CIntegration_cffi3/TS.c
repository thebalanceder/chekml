#include "TS.h"
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#define NEIGHBORHOOD_SIZE 20
#define TABU_TENURE 10
#define TABU_LIST_SIZE 64 // Fixed size for GPU, must be >= dim * 2
#define INF 1e30

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
static const char* ts_kernel_source =
"#define TABU_LIST_SIZE 64\n"
"#define INF 1e30f\n"
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
"__kernel void generate_and_evaluate_neighbors(\n"
"    __global const float* current_solution,\n"
"    __global const float* bounds,\n"
"    __global float* neighbors,\n"
"    __global float* neighbor_fitness,\n"
"    __global int* neighbor_moves,\n"
"    __global const int* tabu_list,\n"
"    const int dim,\n"
"    const int neighborhood_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < neighborhood_size) {\n"
"        uint local_seed = seed + id * 2654435761u;\n"
"\n"
"        // Generate neighbor\n"
"        int move_hash = 0;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float delta = lcg_rand_float(&local_seed, -0.1f, 0.1f);\n"
"            float pos = current_solution[d] + delta;\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            pos = fmax(min, fmin(max, pos));\n"
"            neighbors[id * dim + d] = pos;\n"
"            int move = (delta > 0.0f) ? 1 : -1;\n"
"            neighbor_moves[id * dim + d] = move;\n"
"            move_hash += (move > 0 ? d + 1 : -(d + 1)) * 31;\n"
"        }\n"
"        move_hash = abs(move_hash) % TABU_LIST_SIZE;\n"
"        neighbor_moves[id * dim + dim] = move_hash;\n"
"\n"
"        // Check tabu status\n"
"        float fitness = INF;\n"
"        if (tabu_list[move_hash] == 0) {\n"
"            // Evaluate fitness (placeholder: CUSTOMIZE OBJECTIVE FUNCTION HERE)\n"
"            fitness = 0.0f;\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float x = neighbors[id * dim + d];\n"
"                fitness += x * x;\n"
"            }\n"
"        }\n"
"        neighbor_fitness[id] = fitness;\n"
"    }\n"
"}\n"
"__kernel void find_best_neighbor(\n"
"    __global const float* neighbor_fitness,\n"
"    __global const float* neighbors,\n"
"    __global float* best_neighbor,\n"
"    __global float* best_fitness,\n"
"    __global int* best_move_hash,\n"
"    const int dim,\n"
"    const int neighborhood_size,\n"
"    __local float* local_fitness,\n"
"    __local int* local_indices)\n"
"{\n"
"    int local_id = get_local_id(0);\n"
"    int global_id = get_global_id(0);\n"
"    int local_size = get_local_size(0);\n"
"\n"
"    // Initialize local arrays\n"
"    if (global_id < neighborhood_size) {\n"
"        local_fitness[local_id] = neighbor_fitness[global_id];\n"
"        local_indices[local_id] = global_id;\n"
"    } else {\n"
"        local_fitness[local_id] = INF;\n"
"        local_indices[local_id] = -1;\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    // Perform reduction\n"
"    for (int offset = local_size / 2; offset > 0; offset /= 2) {\n"
"        if (local_id < offset) {\n"
"            if (local_fitness[local_id + offset] < local_fitness[local_id]) {\n"
"                local_fitness[local_id] = local_fitness[local_id + offset];\n"
"                local_indices[local_id] = local_indices[local_id + offset];\n"
"            }\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"\n"
"    // Write result from first thread\n"
"    if (local_id == 0 && get_group_id(0) == 0) {\n"
"        int best_idx = local_indices[0];\n"
"        *best_fitness = local_fitness[0];\n"
"        if (best_idx >= 0 && *best_fitness < INF) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                best_neighbor[d] = neighbors[best_idx * dim + d];\n"
"            }\n"
"            *best_move_hash = neighbors[best_idx * dim + dim];\n"
"        } else {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                best_neighbor[d] = 0.0f;\n"
"            }\n"
"            *best_fitness = INF;\n"
"            *best_move_hash = -1;\n"
"        }\n"
"    }\n"
"}\n";

void TS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel neighbor_kernel = NULL, best_kernel = NULL;
    cl_mem current_solution_buffer = NULL, bounds_buffer = NULL;
    cl_mem neighbors_buffer = NULL, neighbor_fitness_buffer = NULL;
    cl_mem neighbor_moves_buffer = NULL, tabu_list_buffer = NULL;
    cl_mem best_neighbor_buffer = NULL, best_fitness_buffer = NULL;
    cl_mem best_move_hash_buffer = NULL;
    float *current_solution = NULL, *bounds = NULL;
    float *neighbors = NULL, *neighbor_fitness = NULL;
    int *neighbor_moves = NULL, *tabu_list = NULL;
    float *best_neighbor = NULL, *best_fitness = NULL;
    int *best_move_hash = NULL;
    double *temp_solution = NULL;
    cl_event events[2] = {0};
    double start_time, end_time;

    // Validate input
    if (!opt || !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        return;
    }

    const int dim = opt->dim;
    const int max_iter = opt->max_iter;
    const int neighborhood_size = NEIGHBORHOOD_SIZE;

    if (dim < 1 || max_iter < 1) {
        fprintf(stderr, "Error: Invalid dim (%d) or max_iter (%d)\n", dim, max_iter);
        return;
    }

    // Initialize host memory
    current_solution = (float*)malloc(dim * sizeof(float));
    bounds = (float*)malloc(2 * dim * sizeof(float));
    neighbors = (float*)malloc(neighborhood_size * (dim + 1) * sizeof(float)); // +1 for move hash
    neighbor_fitness = (float*)malloc(neighborhood_size * sizeof(float));
    neighbor_moves = (int*)malloc(neighborhood_size * (dim + 1) * sizeof(int)); // +1 for move hash
    tabu_list = (int*)calloc(TABU_LIST_SIZE, sizeof(int));
    best_neighbor = (float*)malloc(dim * sizeof(float));
    best_fitness = (float*)malloc(sizeof(float));
    best_move_hash = (int*)malloc(sizeof(int));
    temp_solution = (double*)malloc(dim * sizeof(double));
    if (!current_solution || !bounds || !neighbors || !neighbor_fitness ||
        !neighbor_moves || !tabu_list || !best_neighbor || !best_fitness ||
        !best_move_hash || !temp_solution) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }

    // Initialize current solution randomly
    srand((unsigned)time(NULL));
    for (int i = 0; i < dim; i++) {
        double min = opt->bounds[2 * i];
        double max = opt->bounds[2 * i + 1];
        current_solution[i] = (float)(min + (max - min) * ((double)rand() / RAND_MAX));
        bounds[2 * i] = (float)min;
        bounds[2 * i + 1] = (float)max;
    }
    for (int i = 0; i < dim; i++) {
        temp_solution[i] = (double)current_solution[i];
    }
    opt->best_solution.fitness = objective_function(temp_solution);
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = temp_solution[i];
    }
    printf("Initial solution: [");
    for (int i = 0; i < dim; i++) {
        printf("%f", current_solution[i]);
        if (i < dim - 1) printf(", ");
    }
    printf("], Fitness: %f\n", opt->best_solution.fitness);

    // Select GPU platform and device
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "Error: No OpenCL platforms found: %d\n", err);
        goto cleanup;
    }
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) {
        fprintf(stderr, "Error: Memory allocation failed for platforms\n");
        goto cleanup;
    }
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error getting platform IDs: %d\n", err);
        free(platforms);
        goto cleanup;
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
            goto cleanup;
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
        goto cleanup;
    }

    // Create context and queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !context) {
        fprintf(stderr, "Error creating OpenCL context: %d\n", err);
        goto cleanup;
    }
    cl_queue_properties queue_props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, queue_props, &err);
    if (err != CL_SUCCESS || !queue) {
        fprintf(stderr, "Error creating OpenCL queue: %d\n", err);
        clReleaseContext(context);
        goto cleanup;
    }

    // Query device capabilities
    size_t max_work_group_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying CL_DEVICE_MAX_WORK_GROUP_SIZE: %d\n", err);
        goto cleanup;
    }
    size_t local_work_size = max_work_group_size < 256 ? max_work_group_size : 256;

    // Create buffers
    current_solution_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    neighbors_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, neighborhood_size * (dim + 1) * sizeof(float), NULL, &err);
    neighbor_fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, neighborhood_size * sizeof(float), NULL, &err);
    neighbor_moves_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, neighborhood_size * (dim + 1) * sizeof(int), NULL, &err);
    tabu_list_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, TABU_LIST_SIZE * sizeof(int), NULL, &err);
    best_neighbor_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    best_fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
    best_move_hash_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || !current_solution_buffer || !bounds_buffer || !neighbors_buffer ||
        !neighbor_fitness_buffer || !neighbor_moves_buffer || !tabu_list_buffer ||
        !best_neighbor_buffer || !best_fitness_buffer || !best_move_hash_buffer) {
        fprintf(stderr, "Error creating buffers: %d\n", err);
        goto cleanup;
    }

    // Write initial data
    err = clEnqueueWriteBuffer(queue, current_solution_buffer, CL_TRUE, 0, dim * sizeof(float), current_solution, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, tabu_list_buffer, CL_TRUE, 0, TABU_LIST_SIZE * sizeof(int), tabu_list, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial buffers: %d\n", err);
        goto cleanup;
    }

    // Create program
    program = clCreateProgramWithSource(context, 1, &ts_kernel_source, NULL, &err);
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
            fprintf(stderr, "Kernel source:\n%s\n", ts_kernel_source);
            free(log);
        }
        goto cleanup;
    }

    // Create kernels
    neighbor_kernel = clCreateKernel(program, "generate_and_evaluate_neighbors", &err);
    best_kernel = clCreateKernel(program, "find_best_neighbor", &err);
    if (err != CL_SUCCESS || !neighbor_kernel || !best_kernel) {
        fprintf(stderr, "Error creating kernels: %d\n", err);
        goto cleanup;
    }

    // Main TS loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Generate and evaluate neighbors
        uint seed = (uint)time(NULL) + iter;
        err = clSetKernelArg(neighbor_kernel, 0, sizeof(cl_mem), &current_solution_buffer);
        err |= clSetKernelArg(neighbor_kernel, 1, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(neighbor_kernel, 2, sizeof(cl_mem), &neighbors_buffer);
        err |= clSetKernelArg(neighbor_kernel, 3, sizeof(cl_mem), &neighbor_fitness_buffer);
        err |= clSetKernelArg(neighbor_kernel, 4, sizeof(cl_mem), &neighbor_moves_buffer);
        err |= clSetKernelArg(neighbor_kernel, 5, sizeof(cl_mem), &tabu_list_buffer);
        err |= clSetKernelArg(neighbor_kernel, 6, sizeof(int), &dim);
        err |= clSetKernelArg(neighbor_kernel, 7, sizeof(int), &neighborhood_size);
        err |= clSetKernelArg(neighbor_kernel, 8, sizeof(uint), &seed);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting neighbor kernel args: %d\n", err);
            goto cleanup;
        }
        size_t global_work_size = ((neighborhood_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, neighbor_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[0]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing neighbor kernel: %d\n", err);
            goto cleanup;
        }

        // Find best neighbor
        err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &neighbor_fitness_buffer);
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &neighbors_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(cl_mem), &best_neighbor_buffer);
        err |= clSetKernelArg(best_kernel, 3, sizeof(cl_mem), &best_fitness_buffer);
        err |= clSetKernelArg(best_kernel, 4, sizeof(cl_mem), &best_move_hash_buffer);
        err |= clSetKernelArg(best_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(best_kernel, 6, sizeof(int), &neighborhood_size);
        err |= clSetKernelArg(best_kernel, 7, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 8, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best kernel args: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing best kernel: %d\n", err);
            goto cleanup;
        }

        // Read best neighbor
        err = clEnqueueReadBuffer(queue, best_neighbor_buffer, CL_TRUE, 0, dim * sizeof(float), best_neighbor, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(queue, best_fitness_buffer, CL_TRUE, 0, sizeof(float), best_fitness, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(queue, best_move_hash_buffer, CL_TRUE, 0, sizeof(int), best_move_hash, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best neighbor data: %d\n", err);
            goto cleanup;
        }

        // Update current solution and tabu list
        if (*best_fitness < INF && *best_move_hash >= 0) {
            // Validate with CPU objective function
            for (int i = 0; i < dim; i++) {
                temp_solution[i] = (double)best_neighbor[i];
            }
            double cpu_fitness = objective_function(temp_solution);
            printf("Iteration %d: GPU Best Fitness = %f, CPU Fitness = %f, Neighbor = [", 
                   iter + 1, *best_fitness, cpu_fitness);
            for (int i = 0; i < dim; i++) {
                printf("%f", best_neighbor[i]);
                if (i < dim - 1) printf(", ");
            }
            printf("]\n");

            if (cpu_fitness < opt->best_solution.fitness || iter == 0) {
                opt->best_solution.fitness = cpu_fitness;
                for (int i = 0; i < dim; i++) {
                    opt->best_solution.position[i] = temp_solution[i];
                }
            }

            memcpy(current_solution, best_neighbor, dim * sizeof(float));
            err = clEnqueueWriteBuffer(queue, current_solution_buffer, CL_TRUE, 0, dim * sizeof(float), current_solution, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing current solution: %d\n", err);
                goto cleanup;
            }

            tabu_list[*best_move_hash] = TABU_TENURE;
            err = clEnqueueWriteBuffer(queue, tabu_list_buffer, CL_TRUE, 0, TABU_LIST_SIZE * sizeof(int), tabu_list, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing tabu list: %d\n", err);
                goto cleanup;
            }
        } else {
            printf("Iteration %d: No valid neighbor found (all tabu or invalid)\n", iter + 1);
        }

        // Update tabu list
        for (int i = 0; i < TABU_LIST_SIZE; i++) {
            if (tabu_list[i] > 0) tabu_list[i]--;
        }
        err = clEnqueueWriteBuffer(queue, tabu_list_buffer, CL_TRUE, 0, TABU_LIST_SIZE * sizeof(int), tabu_list, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error updating tabu list: %d\n", err);
            goto cleanup;
        }

        // Print best solution
        printf("Iteration %d: Best Solution = [", iter + 1);
        for (int i = 0; i < dim; i++) {
            printf("%f", opt->best_solution.position[i]);
            if (i < dim - 1) printf(", ");
        }
        printf("], Fitness = %f\n", opt->best_solution.fitness);

        // Profiling
        cl_ulong time_start, time_end;
        cl_ulong queue_props;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(cl_ulong), &queue_props, NULL);
        if (err == CL_SUCCESS && (queue_props & CL_QUEUE_PROFILING_ENABLE)) {
            for (int i = 0; i < 2; i++) {
                if (events[i]) {
                    err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                    if (err == CL_SUCCESS) {
                        err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                        if (err == CL_SUCCESS) {
                            double time_ms = (time_end - time_start) / 1e6;
                            if (i == 0) printf("  Neighbor Gen/Eval: %.3f ms\n", time_ms);
                            else if (i == 1) printf("  Best Neighbor: %.3f ms\n", time_ms);
                        }
                    }
                    clReleaseEvent(events[i]);
                    events[i] = NULL;
                }
            }
            end_time = get_time_ms();
            printf("TS|%d: Total: %.3f ms\n", iter + 1, end_time - start_time);
        }
    }

    // Validate final solution
    for (int i = 0; i < dim; i++) {
        temp_solution[i] = opt->best_solution.position[i];
    }
    double final_cpu_fitness = objective_function(temp_solution);
    printf("Final solution: [");
    for (int i = 0; i < dim; i++) {
        printf("%f", opt->best_solution.position[i]);
        if (i < dim - 1) printf(", ");
    }
    printf("]\nFinal fitness (CPU): %f\n", final_cpu_fitness);
    if (fabs(final_cpu_fitness - opt->best_solution.fitness) > 1e-5) {
        printf("Warning: Final CPU fitness (%f) differs from stored fitness (%f)\n",
               final_cpu_fitness, opt->best_solution.fitness);
    }
    int is_boundary = 1;
    for (int i = 0; i < dim; i++) {
        double pos = opt->best_solution.position[i];
        double min = opt->bounds[2 * i];
        double max = opt->bounds[2 * i + 1];
        if (fabs(pos - min) > 1e-5 && fabs(pos - max) > 1e-5) {
            is_boundary = 0;
            break;
        }
    }
    if (is_boundary) {
        printf("Warning: Final solution is at bounds, which may be incorrect. "
               "Please provide the correct objective function.\n");
    }
    for (int i = 0; i < dim; i++) {
        double pos = opt->best_solution.position[i];
        double min = opt->bounds[2 * i];
        double max = opt->bounds[2 * i + 1];
        if (pos < min - 1e-5 || pos > max + 1e-5) {
            printf("Warning: Final solution[%d] = %f violates bounds [%f, %f]\n", 
                   i, pos, min, max);
        }
    }

cleanup:
    clFinish(queue);
    if (current_solution) free(current_solution);
    if (bounds) free(bounds);
    if (neighbors) free(neighbors);
    if (neighbor_fitness) free(neighbor_fitness);
    if (neighbor_moves) free(neighbor_moves);
    if (tabu_list) free(tabu_list);
    if (best_neighbor) free(best_neighbor);
    if (best_fitness) free(best_fitness);
    if (best_move_hash) free(best_move_hash);
    if (temp_solution) free(temp_solution);
    if (current_solution_buffer) clReleaseMemObject(current_solution_buffer);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (neighbors_buffer) clReleaseMemObject(neighbors_buffer);
    if (neighbor_fitness_buffer) clReleaseMemObject(neighbor_fitness_buffer);
    if (neighbor_moves_buffer) clReleaseMemObject(neighbor_moves_buffer);
    if (tabu_list_buffer) clReleaseMemObject(tabu_list_buffer);
    if (best_neighbor_buffer) clReleaseMemObject(best_neighbor_buffer);
    if (best_fitness_buffer) clReleaseMemObject(best_fitness_buffer);
    if (best_move_hash_buffer) clReleaseMemObject(best_move_hash_buffer);
    if (neighbor_kernel) clReleaseKernel(neighbor_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}
