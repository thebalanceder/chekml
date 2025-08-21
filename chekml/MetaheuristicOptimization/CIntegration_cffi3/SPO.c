#include "SPO.h"
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Constants
#define MAX_ITER 100
#define PAINT_FACTOR 0.1
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
static const char* spo_kernel_source =
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
"    const int pop_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < pop_size * dim) {\n"
"        int ind = id / dim;\n"
"        int d = id % dim;\n"
"        uint local_seed = seed + id * 2654435761u;\n"
"        float min = bounds[2 * d];\n"
"        float max = bounds[2 * d + 1];\n"
"        population[ind * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"    }\n"
"}\n"
"__kernel void evaluate_population(\n"
"    __global const float* population,\n"
"    __global float* fitness,\n"
"    const int dim,\n"
"    const int pop_size)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < pop_size) {\n"
"        float energy = 0.0f;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float x = population[id * dim + d];\n"
"            energy += x * x; // Placeholder: CUSTOMIZE OBJECTIVE FUNCTION HERE\n"
"        }\n"
"        fitness[id] = energy;\n"
"    }\n"
"}\n"
"__kernel void update_population(\n"
"    __global float* population,\n"
"    __global float* fitness,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    const int n1st,\n"
"    const int n2nd,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < pop_size) {\n"
"        uint local_seed = seed + id * 2654435761u;\n"
"        float new_positions[3 * 4]; // 3 combinations (complement, triangle, rectangle)\n"
"        float new_fitness[3];\n"
"        int best_idx = 0;\n"
"        float best_fitness = fitness[id];\n"
"        int id1, id2, id3, id4;\n"
"\n"
"        // Complement Combination\n"
"        id1 = lcg_rand(&local_seed) % n1st;\n"
"        id2 = lcg_rand(&local_seed) % (pop_size - n1st - n2nd);\n"
"        for (int d = 0; d < dim; d++) {\n"
"            new_positions[0 * dim + d] = population[id * dim + d] +\n"
"                lcg_rand_float(&local_seed, 0.0f, 1.0f) *\n"
"                (population[id1 * dim + d] - population[(n1st + n2nd + id2) * dim + d]);\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            new_positions[0 * dim + d] = new_positions[0 * dim + d] < min ? min :\n"
"                                         new_positions[0 * dim + d] > max ? max :\n"
"                                         new_positions[0 * dim + d];\n"
"        }\n"
"        new_fitness[0] = 0.0f;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float x = new_positions[0 * dim + d];\n"
"            new_fitness[0] += x * x; // Placeholder: CUSTOMIZE OBJECTIVE FUNCTION HERE\n"
"        }\n"
"\n"
"        // Triangle Combination\n"
"        id1 = lcg_rand(&local_seed) % n1st;\n"
"        id2 = lcg_rand(&local_seed) % n2nd;\n"
"        id3 = lcg_rand(&local_seed) % (pop_size - n1st - n2nd);\n"
"        for (int d = 0; d < dim; d++) {\n"
"            new_positions[1 * dim + d] = population[id * dim + d] +\n"
"                lcg_rand_float(&local_seed, 0.0f, 1.0f) *\n"
"                ((population[id1 * dim + d] +\n"
"                  population[(n1st + id2) * dim + d] +\n"
"                  population[(n1st + n2nd + id3) * dim + d]) / 3.0f);\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            new_positions[1 * dim + d] = new_positions[1 * dim + d] < min ? min :\n"
"                                         new_positions[1 * dim + d] > max ? max :\n"
"                                         new_positions[1 * dim + d];\n"
"        }\n"
"        new_fitness[1] = 0.0f;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float x = new_positions[1 * dim + d];\n"
"            new_fitness[1] += x * x; // Placeholder: CUSTOMIZE OBJECTIVE FUNCTION HERE\n"
"        }\n"
"\n"
"        // Rectangle Combination\n"
"        id1 = lcg_rand(&local_seed) % n1st;\n"
"        id2 = lcg_rand(&local_seed) % n2nd;\n"
"        id3 = lcg_rand(&local_seed) % (pop_size - n1st - n2nd);\n"
"        id4 = lcg_rand(&local_seed) % pop_size;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            new_positions[2 * dim + d] = population[id * dim + d] +\n"
"                lcg_rand_float(&local_seed, 0.0f, 1.0f) * population[id1 * dim + d] +\n"
"                lcg_rand_float(&local_seed, 0.0f, 1.0f) * population[(n1st + id2) * dim + d] +\n"
"                lcg_rand_float(&local_seed, 0.0f, 1.0f) * population[(n1st + n2nd + id3) * dim + d] +\n"
"                lcg_rand_float(&local_seed, 0.0f, 1.0f) * population[id4 * dim + d];\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            new_positions[2 * dim + d] = new_positions[2 * dim + d] < min ? min :\n"
"                                         new_positions[2 * dim + d] > max ? max :\n"
"                                         new_positions[2 * dim + d];\n"
"        }\n"
"        new_fitness[2] = 0.0f;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float x = new_positions[2 * dim + d];\n"
"            new_fitness[2] += x * x; // Placeholder: CUSTOMIZE OBJECTIVE FUNCTION HERE\n"
"        }\n"
"\n"
"        // Select best candidate\n"
"        for (int i = 0; i < 3; i++) {\n"
"            if (new_fitness[i] < best_fitness) {\n"
"                best_fitness = new_fitness[i];\n"
"                best_idx = i;\n"
"            }\n"
"        }\n"
"\n"
"        // Update population if better\n"
"        if (best_fitness < fitness[id]) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                population[id * dim + d] = new_positions[best_idx * dim + d];\n"
"            }\n"
"            fitness[id] = best_fitness;\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void find_best_solution(\n"
"    __global const float* fitness,\n"
"    __global const float* population,\n"
"    __global float* best_position,\n"
"    __global float* best_fitness,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    __local float* local_fitness,\n"
"    __local int* local_indices)\n"
"{\n"
"    int local_id = get_local_id(0);\n"
"    int global_id = get_global_id(0);\n"
"    int local_size = get_local_size(0);\n"
"\n"
"    // Initialize local arrays\n"
"    if (global_id < pop_size) {\n"
"        local_fitness[local_id] = fitness[global_id];\n"
"        local_indices[local_id] = global_id;\n"
"    } else {\n"
"        local_fitness[local_id] = INFINITY;\n"
"        local_indices[local_id] = -1;\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    // Perform reduction within work-group\n"
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
"    // Write result from first thread of first work-group\n"
"    if (local_id == 0 && get_group_id(0) == 0) {\n"
"        int best_idx = local_indices[0];\n"
"        *best_fitness = local_fitness[0];\n"
"        if (best_idx >= 0) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                best_position[d] = population[best_idx * dim + d];\n"
"            }\n"
"        } else {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                best_position[d] = 0.0f;\n"
"            }\n"
"            *best_fitness = INFINITY;\n"
"        }\n"
"    }\n"
"}\n";

void SPO_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, eval_kernel = NULL, update_kernel = NULL, best_kernel = NULL;
    cl_mem population_buffer = NULL, fitness_buffer = NULL, bounds_buffer = NULL;
    cl_mem best_position_buffer = NULL, best_fitness_buffer = NULL;
    float *population = NULL, *fitness = NULL, *bounds = NULL;
    float *best_position = NULL, *best_fitness = NULL;
    double *temp_solution = NULL;
    cl_event events[4] = {0};
    double start_time, end_time;

    // Validate input
    if (!opt || !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        return;
    }

    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    const int n1st = pop_size / 3;
    const int n2nd = pop_size / 3;
    const int n3rd = pop_size - n1st - n2nd;

    if (dim < 1 || pop_size < 3) {
        fprintf(stderr, "Error: Invalid dim (%d) or population_size (%d)\n", dim, pop_size);
        return;
    }

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int d = 0; d < dim; d++) {
        opt->best_solution.position[d] = opt->bounds[2 * d]; // Initialize to lower bound
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
    size_t local_work_size = max_work_group_size < 256 ? max_work_group_size : 256;

    // Allocate host memory
    population = (float*)malloc(pop_size * dim * sizeof(float));
    fitness = (float*)malloc(pop_size * sizeof(float));
    bounds = (float*)malloc(2 * dim * sizeof(float));
    best_position = (float*)malloc(dim * sizeof(float));
    best_fitness = (float*)malloc(sizeof(float));
    temp_solution = (double*)malloc(dim * sizeof(double));
    if (!population || !fitness || !bounds || !best_position || !best_fitness || !temp_solution) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds[i] = (float)opt->bounds[i];
    }

    // Create buffers
    population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * sizeof(float), NULL, &err);
    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    best_position_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    best_fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !population_buffer || !fitness_buffer || !bounds_buffer ||
        !best_position_buffer || !best_fitness_buffer) {
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
    program = clCreateProgramWithSource(context, 1, &spo_kernel_source, NULL, &err);
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
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    eval_kernel = clCreateKernel(program, "evaluate_population", &err);
    update_kernel = clCreateKernel(program, "update_population", &err);
    best_kernel = clCreateKernel(program, "find_best_solution", &err);
    if (err != CL_SUCCESS || !init_kernel || !eval_kernel || !update_kernel || !best_kernel) {
        fprintf(stderr, "Error creating kernels: %d\n", err);
        goto cleanup;
    }

    // Initialize population
    uint seed = (uint)time(NULL);
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 3, sizeof(int), &pop_size);
    err |= clSetKernelArg(init_kernel, 4, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting init kernel args: %d\n", err);
        goto cleanup;
    }
    size_t init_global_work_size = ((pop_size * dim + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, &init_global_work_size, &local_work_size, 0, NULL, &events[0]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel: %d\n", err);
        goto cleanup;
    }

    // Evaluate initial population
    err = clSetKernelArg(eval_kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(eval_kernel, 1, sizeof(cl_mem), &fitness_buffer);
    err |= clSetKernelArg(eval_kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(eval_kernel, 3, sizeof(int), &pop_size);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting eval kernel args: %d\n", err);
        goto cleanup;
    }
    size_t eval_global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, eval_kernel, 1, NULL, &eval_global_work_size, &local_work_size, 0, NULL, &events[1]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing eval kernel: %d\n", err);
        goto cleanup;
    }

    // Debug: Read initial fitness to check population diversity
    err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, 0, pop_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading initial fitness: %d\n", err);
        goto cleanup;
    }
    float min_fitness = INFINITY, max_fitness = -INFINITY;
    for (int i = 0; i < pop_size; i++) {
        if (fitness[i] < min_fitness) min_fitness = fitness[i];
        if (fitness[i] > max_fitness) max_fitness = fitness[i];
    }
    printf("Initial fitness range: [%f, %f]\n", min_fitness, max_fitness);    

    // Main SPO loop
    for (int iter = 0; iter < MAX_ITER; iter++) {
        start_time = get_time_ms();

        // Read population for sorting
        err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, 0, pop_size * dim * sizeof(float), population, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading population: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, 0, pop_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading fitness: %d\n", err);
            goto cleanup;
        }

        // Sort population by fitness (CPU)
        int* indices = (int*)malloc(pop_size * sizeof(int));
        if (!indices) {
            fprintf(stderr, "Error: Memory allocation failed for indices\n");
            goto cleanup;
        }
        for (int i = 0; i < pop_size; i++) indices[i] = i;
        for (int i = 0; i < pop_size - 1; i++) {
            for (int j = i + 1; j < pop_size; j++) {
                if (fitness[indices[i]] > fitness[indices[j]]) {
                    int temp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp;
                }
            }
        }
        float* sorted_population = (float*)malloc(pop_size * dim * sizeof(float));
        float* sorted_fitness = (float*)malloc(pop_size * sizeof(float));
        if (!sorted_population || !sorted_fitness) {
            fprintf(stderr, "Error: Memory allocation failed for sorted arrays\n");
            free(indices); if (sorted_population) free(sorted_population); if (sorted_fitness) free(sorted_fitness);
            goto cleanup;
        }
        for (int i = 0; i < pop_size; i++) {
            sorted_fitness[i] = fitness[indices[i]];
            for (int d = 0; d < dim; d++) {
                sorted_population[i * dim + d] = population[indices[i] * dim + d];
            }
        }
        err = clEnqueueWriteBuffer(queue, population_buffer, CL_TRUE, 0, pop_size * dim * sizeof(float), sorted_population, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing sorted population: %d\n", err);
            free(indices); free(sorted_population); free(sorted_fitness);
            goto cleanup;
        }
        err = clEnqueueWriteBuffer(queue, fitness_buffer, CL_TRUE, 0, pop_size * sizeof(float), sorted_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing sorted fitness: %d\n", err);
            free(indices); free(sorted_population); free(sorted_fitness);
            goto cleanup;
        }
        free(indices); free(sorted_population); free(sorted_fitness);

        // Update population
        seed = (uint)time(NULL) + iter;
        err = clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(update_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(update_kernel, 3, sizeof(int), &dim);
        err |= clSetKernelArg(update_kernel, 4, sizeof(int), &pop_size);
        err |= clSetKernelArg(update_kernel, 5, sizeof(int), &n1st);
        err |= clSetKernelArg(update_kernel, 6, sizeof(int), &n2nd);
        err |= clSetKernelArg(update_kernel, 7, sizeof(uint), &seed);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting update kernel args: %d\n", err);
            goto cleanup;
        }
        size_t update_global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, update_kernel, 1, NULL, &update_global_work_size, &local_work_size, 0, NULL, &events[2]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing update kernel: %d\n", err);
            goto cleanup;
        }

        // Find best solution
        err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(cl_mem), &best_position_buffer);
        err |= clSetKernelArg(best_kernel, 3, sizeof(cl_mem), &best_fitness_buffer);
        err |= clSetKernelArg(best_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(best_kernel, 5, sizeof(int), &pop_size);
        err |= clSetKernelArg(best_kernel, 6, local_work_size * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 7, local_work_size * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best kernel args: %d\n", err);
            goto cleanup;
        }
        size_t best_global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &best_global_work_size, &local_work_size, 0, NULL, &events[3]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing best kernel: %d\n", err);
            goto cleanup;
        }

        // Read best solution
        err = clEnqueueReadBuffer(queue, best_fitness_buffer, CL_TRUE, 0, sizeof(float), best_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best fitness: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueReadBuffer(queue, best_position_buffer, CL_TRUE, 0, dim * sizeof(float), best_position, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best position: %d\n", err);
            goto cleanup;
        }

        // Debug: Print GPU best
        printf("Iteration %d: GPU Best Fitness = %f, Position = [", iter + 1, *best_fitness);
        for (int d = 0; d < dim; d++) {
            printf("%f", best_position[d]);
            if (d < dim - 1) printf(", ");
        }
        printf("]\n");

        // Update best solution with CPU validation
        if (*best_fitness < INFINITY) {
            for (int d = 0; d < dim; d++) {
                temp_solution[d] = (double)best_position[d];
            }
            double cpu_fitness = objective_function(temp_solution);
            if (cpu_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = cpu_fitness;
                for (int d = 0; d < dim; d++) {
                    opt->best_solution.position[d] = temp_solution[d];
                }
            }
        }

        // Enforce bounds
        enforce_bound_constraints(opt);

        // Print progress
        printf("Iteration %d: Best Fitness (CPU) = %f, Position = [", iter + 1, opt->best_solution.fitness);
        for (int d = 0; d < dim; d++) {
            printf("%f", opt->best_solution.position[d]);
            if (d < dim - 1) printf(", ");
        }
        printf("]\n");

        // Profiling
        cl_ulong time_start, time_end;
        cl_ulong queue_props;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(cl_ulong), &queue_props, NULL);
        if (err == CL_SUCCESS && (queue_props & CL_QUEUE_PROFILING_ENABLE)) {
            for (int i = 0; i < 4; i++) {
                if (events[i]) {
                    err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                    if (err == CL_SUCCESS) {
                        err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                        if (err == CL_SUCCESS) {
                            double time_ms = (time_end - time_start) / 1e6;
                            if (i == 0) printf("  Init: %.3f ms\n", time_ms);
                            else if (i == 1) printf("  Eval: %.3f ms\n", time_ms);
                            else if (i == 2) printf("  Update: %.3f ms\n", time_ms);
                            else if (i == 3) printf("  Best: %.3f ms\n", time_ms);
                        }
                    }
                    clReleaseEvent(events[i]);
                    events[i] = NULL;
                }
            }
            end_time = get_time_ms();
            printf("SPO|%d: Total: %.3f ms\n", iter + 1, end_time - start_time);
        }
    }

    // Validate final solution
    for (int d = 0; d < dim; d++) {
        temp_solution[d] = opt->best_solution.position[d];
    }
    double final_cpu_fitness = objective_function(temp_solution);
    printf("Final solution: [");
    for (int d = 0; d < dim; d++) {
        printf("%f", opt->best_solution.position[d]);
        if (d < dim - 1) printf(", ");
    }
    printf("]\nFinal fitness (CPU): %f\n", final_cpu_fitness);
    if (fabs(final_cpu_fitness - opt->best_solution.fitness) > 1e-5) {
        printf("Warning: Final CPU fitness (%f) differs from stored fitness (%f)\n",
               final_cpu_fitness, opt->best_solution.fitness);
    }
    bool is_zero = true;
    for (int d = 0; d < dim; d++) {
        if (fabs(opt->best_solution.position[d]) > 1e-5) {
            is_zero = false;
            break;
        }
    }
    if (is_zero) {
        printf("Warning: Final solution is [0, 0], which may be incorrect. "
               "Please provide the correct objective function.\n");
    }

    // Check bounds
    for (int d = 0; d < dim; d++) {
        double pos = opt->best_solution.position[d];
        double min = opt->bounds[2 * d];
        double max = opt->bounds[2 * d + 1];
        if (pos < min - 1e-5 || pos > max + 1e-5) {
            printf("Warning: Final solution[%d] = %f violates bounds [%f, %f]\n", d, pos, min, max);
        }
    }

cleanup:
    clFinish(queue);
    if (population) free(population);
    if (fitness) free(fitness);
    if (bounds) free(bounds);
    if (best_position) free(best_position);
    if (best_fitness) free(best_fitness);
    if (temp_solution) free(temp_solution);
    if (population_buffer) clReleaseMemObject(population_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (best_position_buffer) clReleaseMemObject(best_position_buffer);
    if (best_fitness_buffer) clReleaseMemObject(best_fitness_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (eval_kernel) clReleaseKernel(eval_kernel);
    if (update_kernel) clReleaseKernel(update_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}
