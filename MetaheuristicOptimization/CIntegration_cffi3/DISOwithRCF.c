/* DISOwithRCF.c - GPU-Optimized DISO Optimization with River Current Flow */
#include "DISOwithRCF.h"
#include <string.h>
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

// OpenCL kernel source
static const char* diso_kernel_source =
"#define DIVERSION_FACTOR 0.3f\n"
"#define FLOW_ADJUSTMENT 0.2f\n"
"#define WATER_DENSITY 1.35f\n"
"#define FLUID_DISTRIBUTION 0.46f\n"
"#define CENTRIFUGAL_RESISTANCE 1.2f\n"
"#define BOTTLENECK_RATIO 0.68f\n"
"#define ELIMINATION_RATIO 0.23f\n"
"#define HRO 1.2f\n"
"#define HRI 7.2f\n"
"#define HGO 1.3f\n"
"#define HGI 0.82f\n"
"#define CFR_FACTOR 9.435f\n"
// Simple LCG random number generator
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
// Enforce bound constraints
"void enforce_bounds(__global float* position, __global const float* bounds, int dim) {\n"
"    for (int d = 0; d < dim; d++) {\n"
"        float min = bounds[2 * d];\n"
"        float max = bounds[2 * d + 1];\n"
"        position[d] = fmax(min, fmin(max, position[d]));\n"
"    }\n"
"}\n"
// Diversion Phase Kernel
"__kernel void diversion_phase(\n"
"    __global float* population,\n"
"    __global const float* best_solution,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        float r1 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float r2 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float CFR = CFR_FACTOR * lcg_rand_float(&local_seed, 0.0f, 1.0f) * 2.5f;\n"
"        float velocity = (r1 < 0.23f) ? (pow(HRO, 2.0f / 3.0f) * sqrt(HGO) / CFR) * r1\n"
"                                     : (pow(HRI, 2.0f / 3.0f) * sqrt(HGI) / CFR) * r2;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float delta = velocity * (best_solution[d] - population[id * dim + d]) * lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"            population[id * dim + d] += delta;\n"
"        }\n"
"        enforce_bounds(&population[id * dim], bounds, dim);\n"
"    }\n"
"}\n"
// Spiral Motion Update Kernel
"__kernel void spiral_motion_update(\n"
"    __global float* population,\n"
"    __global const float* fitness,\n"
"    __global const float* best_solution,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int iter,\n"
"    const int max_iter,\n"
"    uint seed,\n"
"    __local float* local_fitness)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < population_size) {\n"
"        local_fitness[local_id] = fitness[id];\n"
"    } else {\n"
"        local_fitness[local_id] = 0.0f;\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    float total_fitness = 0.0f;\n"
"    for (int i = 0; i < get_local_size(0); i++) {\n"
"        total_fitness += local_fitness[i];\n"
"    }\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        float MLV = total_fitness / population_size;\n"
"        float LP = (WATER_DENSITY * FLUID_DISTRIBUTION * MLV * MLV) / CENTRIFUGAL_RESISTANCE;\n"
"        float RCF = WATER_DENSITY * cos(radians(90.0f * ((float)iter / max_iter))) * \n"
"                    sqrt(pow(best_solution[dim] - fitness[id], 2.0f));\n"
"        if (RCF > LP) {\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float min = bounds[2 * d];\n"
"                float max = bounds[2 * d + 1];\n"
"                population[id * dim + d] = min + lcg_rand_float(&local_seed, 0.0f, 1.0f) * (max - min);\n"
"            }\n"
"        }\n"
"        enforce_bounds(&population[id * dim], bounds, dim);\n"
"    }\n"
"}\n"
// Local Development Phase Kernel
"__kernel void local_development_phase(\n"
"    __global float* population,\n"
"    __global const float* best_solution,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        float r3 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        float CFR = CFR_FACTOR * lcg_rand_float(&local_seed, 0.0f, 1.0f) * 2.5f;\n"
"        float velocity = (pow(HRI, 2.0f / 3.0f) * sqrt(HGI) / (2.0f * CFR)) * \n"
"                         ((r3 < BOTTLENECK_RATIO) ? r3 : lcg_rand_float(&local_seed, 0.0f, 1.0f));\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float delta = velocity * (best_solution[d] - population[id * dim + d]) * lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"            population[id * dim + d] += delta;\n"
"        }\n"
"        enforce_bounds(&population[id * dim], bounds, dim);\n"
"    }\n"
"}\n"
// Elimination Phase Kernel
"__kernel void elimination_phase(\n"
"    __global float* population,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int worst_count = (int)(ELIMINATION_RATIO * population_size);\n"
"    if (id >= population_size - worst_count && id < population_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            population[id * dim + d] = min + lcg_rand_float(&local_seed, 0.0f, 1.0f) * (max - min);\n"
"        }\n"
"    }\n"
"}\n"
// Find Best Solution Kernel
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

// Phase function implementations
void diversion_phase(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, cl_mem best_solution_buffer, 
                     cl_mem bounds_buffer, int dim, int population_size, uint seed, cl_event *event) {
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &best_solution_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &dim);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &population_size);
    err |= clSetKernelArg(kernel, 5, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting diversion kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing diversion kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

void spiral_motion_update(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, cl_mem fitness_buffer, 
                         cl_mem best_solution_buffer, cl_mem bounds_buffer, int dim, int population_size, int iter, 
                         int max_iter, uint seed, cl_event *event) {
    cl_int err;
    size_t local_work_size = 64; // Declare local_work_size here
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &fitness_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &best_solution_buffer);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &dim);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &population_size);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &iter);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &max_iter);
    err |= clSetKernelArg(kernel, 8, sizeof(uint), &seed);
    err |= clSetKernelArg(kernel, 9, local_work_size * sizeof(float), NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting spiral motion kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing spiral motion kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

void local_development_phase(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                            cl_mem best_solution_buffer, cl_mem bounds_buffer, int dim, int population_size, 
                            uint seed, cl_event *event) {
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &best_solution_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &dim);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &population_size);
    err |= clSetKernelArg(kernel, 5, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting local development kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing local development kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

void elimination_phase(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, cl_mem bounds_buffer, 
                       int dim, int population_size, uint seed, cl_event *event) {
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &population_size);
    err |= clSetKernelArg(kernel, 4, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting elimination kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing elimination kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

// Main Optimization Function
void DISO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel diversion_kernel = NULL, spiral_kernel = NULL, local_kernel = NULL, elimination_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, best_solution_buffer = NULL, best_idx_buffer = NULL;
    float *bounds_float = NULL, *population = NULL, *fitness = NULL, *best_solution = NULL;
    int *best_idx_array = NULL;
    double *temp_position = NULL;
    cl_event events[5] = {0};
    double start_time, end_time;

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
    if (dim < 1 || population_size < 1 || max_iter < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), or max_iter (%d)\n", 
                dim, population_size, max_iter);
        exit(EXIT_FAILURE);
    }

    // Select GPU platform and device
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
        platform = platforms[i];
        device = devices[0];
        free(devices);
        break;
    }
    free(platforms);
    if (!platform || !device) {
        fprintf(stderr, "Error: No GPU device found\n");
        exit(EXIT_FAILURE);
    }

    // Create context and queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating OpenCL context: %d\n", err);
        exit(EXIT_FAILURE);
    }
    cl_queue_properties queue_props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, queue_props, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating OpenCL queue: %d\n", err);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }

    // Create program
    program = clCreateProgramWithSource(context, 1, &diso_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating DISO program: %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building DISO program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    diversion_kernel = clCreateKernel(program, "diversion_phase", &err);
    spiral_kernel = clCreateKernel(program, "spiral_motion_update", &err);
    local_kernel = clCreateKernel(program, "local_development_phase", &err);
    elimination_kernel = clCreateKernel(program, "elimination_phase", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !diversion_kernel || !spiral_kernel || !local_kernel || !elimination_kernel || !best_kernel) {
        fprintf(stderr, "Error creating DISO kernels: %d\n", err);
        goto cleanup;
    }

    // Create buffers
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    population = (float*)malloc(population_size * dim * sizeof(float));
    fitness = (float*)malloc(population_size * sizeof(float));
    best_solution = (float*)malloc((dim + 1) * sizeof(float)); // +1 for fitness
    best_idx_array = (int*)malloc(population_size * sizeof(int));
    temp_position = (double*)malloc(dim * sizeof(double));
    if (!bounds_float || !population || !fitness || !best_solution || !best_idx_array || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < dim; i++) {
        best_solution[i] = 0.0f;
    }
    best_solution[dim] = INFINITY;

    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    best_solution_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (dim + 1) * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    opt->population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    opt->fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !best_solution_buffer || !best_idx_buffer || 
        !opt->population_buffer || !opt->fitness_buffer) {
        fprintf(stderr, "Error creating DISO buffers: %d\n", err);
        goto cleanup;
    }

    // Write initial data
    err = clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, best_solution_buffer, CL_TRUE, 0, (dim + 1) * sizeof(float), best_solution, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial buffers: %d\n", err);
        goto cleanup;
    }

    // Initialize population on host (could be moved to GPU if needed)
    uint seed = (uint)time(NULL);
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            float min = bounds_float[2 * d];
            float max = bounds_float[2 * d + 1];
            population[i * dim + d] = min + (max - min) * ((float)rand() / RAND_MAX);
        }
    }
    err = clEnqueueWriteBuffer(queue, opt->population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing population buffer: %d\n", err);
        goto cleanup;
    }

    // Evaluate initial population fitness on host
    err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading population buffer: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            temp_position[d] = (double)population[i * dim + d];
        }
        fitness[i] = (float)objective_function(temp_position);
    }
    err = clEnqueueWriteBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing fitness buffer: %d\n", err);
        goto cleanup;
    }

    // Main loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Execute phases
        diversion_phase(diversion_kernel, queue, opt->population_buffer, best_solution_buffer, bounds_buffer, 
                        dim, population_size, seed, &events[0]);
        spiral_motion_update(spiral_kernel, queue, opt->population_buffer, opt->fitness_buffer, best_solution_buffer, 
                            bounds_buffer, dim, population_size, iter, max_iter, seed, &events[1]);
        local_development_phase(local_kernel, queue, opt->population_buffer, best_solution_buffer, bounds_buffer, 
                               dim, population_size, seed, &events[2]);
        elimination_phase(elimination_kernel, queue, opt->population_buffer, bounds_buffer, 
                         dim, population_size, seed, &events[3]);

        // Evaluate population fitness on host
        err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading population buffer: %d\n", err);
            goto cleanup;
        }
        for (int i = 0; i < population_size; i++) {
            for (int d = 0; d < dim; d++) {
                temp_position[d] = (double)population[i * dim + d];
            }
            fitness[i] = (float)objective_function(temp_position);
        }
        err = clEnqueueWriteBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            goto cleanup;
        }

        // Find best solution
        err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &best_idx_buffer);
        err |= clSetKernelArg(best_kernel, 2, sizeof(int), &population_size);
        err |= clSetKernelArg(best_kernel, 3, 64 * sizeof(float), NULL);
        err |= clSetKernelArg(best_kernel, 4, 64 * sizeof(int), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting best kernel args: %d\n", err);
            goto cleanup;
        }
        size_t local_work_size = 64;
        size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
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
        float best_fitness = fitness[best_idx];
        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)best_fitness;
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)population[best_idx * dim + d];
                best_solution[d] = population[best_idx * dim + d];
            }
            best_solution[dim] = best_fitness;
            err = clEnqueueWriteBuffer(queue, best_solution_buffer, CL_TRUE, 0, (dim + 1) * sizeof(float), best_solution, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing best_solution buffer: %d\n", err);
                goto cleanup;
            }
        }

        // Profiling
        cl_ulong time_start, time_end;
        double diversion_time = 0, spiral_time = 0, local_time = 0, elimination_time = 0, best_time = 0;
        for (int i = 0; i < 5; i++) {
            if (events[i]) {
                err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                    if (err == CL_SUCCESS) {
                        double time_ms = (time_end - time_start) / 1e6;
                        if (i == 0) diversion_time = time_ms;
                        else if (i == 1) spiral_time = time_ms;
                        else if (i == 2) local_time = time_ms;
                        else if (i == 3) elimination_time = time_ms;
                        else best_time = time_ms;
                    }
                }
            }
        }
        end_time = get_time_ms();
        printf("DISO|%5d -----> %9.16f | Total: %.3f ms | Diversion: %.3f ms | Spiral: %.3f ms | Local: %.3f ms | Elimination: %.3f ms | Best: %.3f ms\n",
               iter + 1, opt->best_solution.fitness, end_time - start_time, 
               diversion_time, spiral_time, local_time, elimination_time, best_time);
    }

    // Update CPU-side population
    err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final population buffer: %d\n", err);
        goto cleanup;
    }
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            opt->population[i].position[d] = (double)population[i * dim + d];
        }
        opt->population[i].fitness = (double)fitness[i];
    }

cleanup:
    if (bounds_float) free(bounds_float);
    if (population) free(population);
    if (fitness) free(fitness);
    if (best_solution) free(best_solution);
    if (best_idx_array) free(best_idx_array);
    if (temp_position) free(temp_position);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (best_solution_buffer) clReleaseMemObject(best_solution_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (opt->population_buffer) clReleaseMemObject(opt->population_buffer);
    if (opt->fitness_buffer) clReleaseMemObject(opt->fitness_buffer);
    if (diversion_kernel) clReleaseKernel(diversion_kernel);
    if (spiral_kernel) clReleaseKernel(spiral_kernel);
    if (local_kernel) clReleaseKernel(local_kernel);
    if (elimination_kernel) clReleaseKernel(elimination_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 5; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (queue) clFinish(queue);
}
