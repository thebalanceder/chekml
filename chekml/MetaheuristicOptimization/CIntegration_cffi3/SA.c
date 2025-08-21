#include "SA.h"
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

// OpenCL kernel source (no evaluation kernel since CPU is used)
static const char* sa_kernel_source =
"#define CANDIDATE_COUNT 1024\n"
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
"        uint local_seed = seed + id * 2654435761u;\n"
"        float min = bounds[2 * id];\n"
"        float max = bounds[2 * id + 1];\n"
"        solution[id] = min + (max - min) * lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"    }\n"
"}\n"
"__kernel void generate_candidates(\n"
"    __global const float* current_solution,\n"
"    __global float* candidates,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    uint seed,\n"
"    float temp)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int candidate_idx = id / dim;\n"
"    int dim_idx = id % dim;\n"
"    if (candidate_idx < CANDIDATE_COUNT && dim_idx < dim) {\n"
"        uint local_seed = seed + id * 2654435761u;\n"
"        float scale = max(temp * 1.0f, 0.1f); // Larger, temperature-scaled perturbation\n"
"        float perturb = lcg_rand_float(&local_seed, -scale, scale);\n"
"        float pos = current_solution[dim_idx] + perturb;\n"
"        float min = bounds[2 * dim_idx];\n"
"        float max = bounds[2 * dim_idx + 1];\n"
"        candidates[candidate_idx * dim + dim_idx] = pos < min ? min : (pos > max ? max : pos);\n"
"    }\n"
"}\n";

void SA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, candidate_kernel = NULL;
    cl_mem solution_buffer = NULL, candidates_buffer = NULL;
    cl_mem bounds_buffer = NULL;
    float *current_solution = NULL, *candidates = NULL, *bounds = NULL;
    double *temp_solution = NULL;
    cl_event events[2] = {0};
    double start_time, end_time;

    // Validate input
    if (!opt || !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        return;
    }

    const int dim = opt->dim;
    const int candidate_count = CANDIDATE_COUNT;

    if (dim < 1 || candidate_count < 1) {
        fprintf(stderr, "Error: Invalid dim (%d) or candidate_count (%d)\n", dim, candidate_count);
        return;
    }

    // Select GPU platform and device
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
            return;
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

    // Allocate host memory
    current_solution = (float*)malloc(dim * sizeof(float));
    candidates = (float*)malloc(candidate_count * dim * sizeof(float));
    bounds = (float*)malloc(2 * dim * sizeof(float));
    temp_solution = (double*)malloc(dim * sizeof(double));
    if (!current_solution || !candidates || !bounds || !temp_solution) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds[i] = (float)opt->bounds[i];
    }

    // Create buffers
    solution_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    candidates_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, candidate_count * dim * sizeof(float), NULL, &err);
    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !solution_buffer || !candidates_buffer || !bounds_buffer) {
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
    program = clCreateProgramWithSource(context, 1, &sa_kernel_source, NULL, &err);
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
    candidate_kernel = clCreateKernel(program, "generate_candidates", &err);
    if (err != CL_SUCCESS || !init_kernel || !candidate_kernel) {
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
    double current_energy = objective_function(temp_solution);
    printf("Initial energy (CPU): %f\n", current_energy);
    opt->best_solution.fitness = current_energy;
    for (int j = 0; j < dim; j++) {
        opt->best_solution.position[j] = (double)current_solution[j];
    }

    // SA loop
    double T = INITIAL_TEMP;
    int itry = 0, success = 0, consec_rej = 0, total_iter = 0;
    int accept_count = 0;
    srand((unsigned int)time(NULL));

    while (1) {
        start_time = get_time_ms();
        itry++;

        // Cooling or stopping
        if (itry >= MAX_TRIES || success >= MAX_SUCCESS) {
            if (T < STOP_TEMP || consec_rej >= MAX_CONSEC_REJ || total_iter >= MAX_ITER_SA) {
                printf("Stopping: T = %e, consec_rej = %d, total_iter = %d\n", T, consec_rej, total_iter);
                break;
            } else {
                T *= COOLING_FACTOR;
                printf("Cooling: T = %e\n", T);
                itry = 0;
                success = 0;
            }
        }

        // Check max iterations
        if (total_iter >= MAX_ITER_SA) {
            printf("Stopping: Reached MAX_ITER_SA = %d\n", MAX_ITER_SA);
            break;
        }

        // Generate candidates on GPU
        err = clSetKernelArg(candidate_kernel, 0, sizeof(cl_mem), &solution_buffer);
        err |= clSetKernelArg(candidate_kernel, 1, sizeof(cl_mem), &candidates_buffer);
        err |= clSetKernelArg(candidate_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(candidate_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(candidate_kernel, 4, sizeof(uint), &seed);
        err |= clSetKernelArg(candidate_kernel, 5, sizeof(float), &T);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting candidate kernel args: %d\n", err);
            goto cleanup;
        }
        size_t candidate_global_work_size = ((candidate_count * dim + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, candidate_kernel, 1, NULL, &candidate_global_work_size, &local_work_size, 0, NULL, &events[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing candidate kernel: %d\n", err);
            goto cleanup;
        }

        // Read candidates
        err = clEnqueueReadBuffer(queue, candidates_buffer, CL_TRUE, 0, candidate_count * dim * sizeof(float), candidates, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading candidates: %d\n", err);
            goto cleanup;
        }

        // Evaluate candidates on CPU
        double best_energy = INFINITY;
        int best_idx = -1;
        float min_coord = INFINITY, max_coord = -INFINITY;
        for (int i = 0; i < candidate_count; i++) {
            for (int j = 0; j < dim; j++) {
                temp_solution[j] = (double)candidates[i * dim + j];
                if (candidates[i * dim + j] < min_coord) min_coord = candidates[i * dim + j];
                if (candidates[i * dim + j] > max_coord) max_coord = candidates[i * dim + j];
            }
            double energy = objective_function(temp_solution);
            if (energy < best_energy) {
                best_energy = energy;
                best_idx = i;
            }
        }

        // Validate best_idx
        if (best_idx < 0 || best_idx >= candidate_count) {
            printf("Warning: Invalid best_idx (%d), skipping iteration %d\n", best_idx, total_iter + 1);
            continue;
        }

        // Print candidate diversity
        printf("Iteration %d: T = %e, Best energy = %f, Candidate range = [%f, %f]\n", 
               total_iter + 1, T, best_energy, min_coord, max_coord);

        // Metropolis criterion
        double delta = current_energy - best_energy;
        if (delta > MIN_DELTA || ((double)rand() / RAND_MAX) < exp(delta / (BOLTZMANN_CONST * T))) {
            for (int j = 0; j < dim; j++) {
                current_solution[j] = candidates[best_idx * dim + j];
            }
            err = clEnqueueWriteBuffer(queue, solution_buffer, CL_TRUE, 0, dim * sizeof(float), current_solution, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing current solution: %d\n", err);
                goto cleanup;
            }
            current_energy = best_energy;
            success++;
            consec_rej = 0;
            accept_count++;
            printf("  Accepted: Energy = %f, Solution = [", current_energy);
            for (int j = 0; j < dim; j++) {
                printf("%f", current_solution[j]);
                if (j < dim - 1) printf(", ");
            }
            printf("]\n");
        } else {
            consec_rej++;
            printf("  Rejected: Delta = %f, Prob = %f\n", delta, exp(delta / (BOLTZMANN_CONST * T)));
        }

        // Update global best
        if (best_energy < opt->best_solution.fitness) {
            opt->best_solution.fitness = best_energy;
            for (int j = 0; j < dim; j++) {
                opt->best_solution.position[j] = (double)candidates[best_idx * dim + j];
            }
            printf("  New global best: Fitness = %f, Solution = [", opt->best_solution.fitness);
            for (int j = 0; j < dim; j++) {
                printf("%f", opt->best_solution.position[j]);
                if (j < dim - 1) printf(", ");
            }
            printf("]\n");
        }

        total_iter++;

        // Print acceptance rate periodically
        if (total_iter % 100 == 0) {
            printf("  Acceptance rate: %.2f%% (%d/%d)\n", 
                   (float)accept_count / total_iter * 100.0f, accept_count, total_iter);
        }

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
                            if (i == 0) printf("  Init: %.3f ms\n", time_ms);
                            else if (i == 1) printf("  Candidate: %.3f ms\n", time_ms);
                        }
                    }
                    clReleaseEvent(events[i]);
                    events[i] = NULL;
                }
            }
            end_time = get_time_ms();
            printf("SA|%d: Total: %.3f ms\n", total_iter, end_time - start_time);
        }
    }

    // Validate final solution
    for (int j = 0; j < dim; j++) {
        temp_solution[j] = opt->best_solution.position[j];
    }
    double final_cpu_energy = objective_function(temp_solution);
    printf("Final solution: [");
    for (int j = 0; j < dim; j++) {
        printf("%f", opt->best_solution.position[j]);
        if (j < dim - 1) printf(", ");
    }
    printf("]\nFinal fitness (CPU): %f\n", final_cpu_energy);
    if (fabs(final_cpu_energy - opt->best_solution.fitness) > 1e-5) {
        printf("Warning: Final CPU fitness (%f) differs from stored fitness (%f)\n",
               final_cpu_energy, opt->best_solution.fitness);
    }

    // Check bounds
    for (int j = 0; j < dim; j++) {
        double pos = opt->best_solution.position[j];
        double min = opt->bounds[2 * j];
        double max = opt->bounds[2 * j + 1];
        if (pos < min - 1e-5 || pos > max + 1e-5) {
            printf("Warning: Final solution[%d] = %f violates bounds [%f, %f]\n", j, pos, min, max);
        }
    }

cleanup:
    clFinish(queue);
    if (current_solution) free(current_solution);
    if (candidates) free(candidates);
    if (bounds) free(bounds);
    if (temp_solution) free(temp_solution);
    if (solution_buffer) clReleaseMemObject(solution_buffer);
    if (candidates_buffer) clReleaseMemObject(candidates_buffer);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (candidate_kernel) clReleaseKernel(candidate_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 2; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}
