/* ES.c - GPU-Optimized Eagle Strategy Optimization */
#include "ES.h"
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
static const char* es_kernel_source =
"#define C1 2.0f\n"
"#define C2 2.0f\n"
"#define W_MAX 0.9f\n"
"#define W_MIN 0.4f\n"
"#define LEVY_BETA 1.5f\n"
"#define LEVY_STEP_SCALE 0.1f\n"
"#define LEVY_PROBABILITY 0.2f\n"
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
// Initialize Kernel
"__kernel void initialize_es(\n"
"    __global float* population,\n"
"    __global float* velocity,\n"
"    __global float* local_best,\n"
"    __global float* local_best_cost,\n"
"    __global float* global_best,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            velocity[id * dim + d] = lcg_rand_float(&local_seed, -1.0f, 1.0f);\n"
"            population[id * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"            local_best[id * dim + d] = population[id * dim + d];\n"
"        }\n"
"        local_best_cost[id] = INFINITY;\n"
"    }\n"
"}\n"
// Update Velocity and Position Kernel
"__kernel void update_velocity_and_position(\n"
"    __global float* population,\n"
"    __global float* velocity,\n"
"    __global const float* local_best,\n"
"    __global const float* global_best,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const int iter,\n"
"    const int max_iter,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        float w = W_MAX - ((W_MAX - W_MIN) * iter / (float)max_iter);\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float r1 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"            float r2 = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"            float pos = population[id * dim + d];\n"
"            float v = velocity[id * dim + d];\n"
"            float pbest = local_best[id * dim + d];\n"
"            float gbest = global_best[d];\n"
"            float cog = C1 * r1 * (pbest - pos);\n"
"            float soc = C2 * r2 * (gbest - pos);\n"
"            v = w * v + cog + soc;\n"
"            pos += v;\n"
"            velocity[id * dim + d] = v;\n"
"            population[id * dim + d] = pos;\n"
"        }\n"
"        enforce_bounds(&population[id * dim], bounds, dim);\n"
"    }\n"
"}\n"
// Lévy Flight Kernel
"__kernel void levy_flight(\n"
"    __global float* population,\n"
"    __global const float* local_best,\n"
"    __global const float* global_best,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        if (lcg_rand_float(&local_seed, 0.0f, 1.0f) < LEVY_PROBABILITY) {\n"
"            float beta = LEVY_BETA;\n"
"            float sigma = pow((tgamma(1.0f + beta) * sin(3.1415926535f * beta / 2.0f)) /\n"
"                              (tgamma((1.0f + beta) / 2.0f) * beta * pow(2.0f, (beta - 1.0f) / 2.0f)), 1.0f / beta);\n"
"            for (int d = 0; d < dim; d++) {\n"
"                float u = lcg_rand_float(&local_seed, 0.0f, 1.0f) * sigma;\n"
"                float v = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"                float step = u / pow(fabs(v), 1.0f / beta);\n"
"                float pbest = local_best[id * dim + d];\n"
"                float gbest = global_best[d];\n"
"                float new_pos = pbest + LEVY_STEP_SCALE * step * (pbest - gbest);\n"
"                new_pos = fmax(bounds[2 * d], fmin(new_pos, bounds[2 * d + 1]));\n"
"                population[id * dim + d] = new_pos;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
// Find Best Kernel
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
void initialize_es(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, cl_mem velocity_buffer, 
                   cl_mem local_best_buffer, cl_mem local_best_cost_buffer, cl_mem global_best_buffer, 
                   cl_mem bounds_buffer, int dim, int population_size, uint seed, cl_event *event) {
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &velocity_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &local_best_buffer);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &local_best_cost_buffer);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &global_best_buffer);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &dim);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &population_size);
    err |= clSetKernelArg(kernel, 8, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting initialize kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing initialize kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

void update_velocity_and_position(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                                 cl_mem velocity_buffer, cl_mem local_best_buffer, cl_mem global_best_buffer, 
                                 cl_mem bounds_buffer, int dim, int population_size, int iter, int max_iter, 
                                 uint seed, cl_event *event) {
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &velocity_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &local_best_buffer);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &global_best_buffer);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &dim);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &population_size);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &iter);
    err |= clSetKernelArg(kernel, 8, sizeof(int), &max_iter);
    err |= clSetKernelArg(kernel, 9, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting update velocity kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing update velocity kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

void levy_flight(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, cl_mem local_best_buffer, 
                 cl_mem global_best_buffer, cl_mem bounds_buffer, int dim, int population_size, uint seed, 
                 cl_event *event) {
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &local_best_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &global_best_buffer);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &dim);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &population_size);
    err |= clSetKernelArg(kernel, 6, sizeof(uint), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting levy flight kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing levy flight kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

void find_best(cl_kernel kernel, cl_command_queue queue, cl_mem fitness_buffer, cl_mem best_idx_buffer, 
               int population_size, cl_event *event) {
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &fitness_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &best_idx_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &population_size);
    err |= clSetKernelArg(kernel, 3, 64 * sizeof(float), NULL);
    err |= clSetKernelArg(kernel, 4, 64 * sizeof(int), NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting find best kernel args: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = 64;
    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing find best kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

// Main Optimization Function
void ES_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, update_kernel = NULL, levy_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, velocity_buffer = NULL, local_best_buffer = NULL, local_best_cost_buffer = NULL;
    cl_mem global_best_buffer = NULL, best_idx_buffer = NULL;
    float *bounds_float = NULL, *population = NULL, *fitness = NULL, *global_best = NULL;
    int *best_idx_array = NULL;
    double *temp_position = NULL;
    cl_event events[4] = {0};
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
    program = clCreateProgramWithSource(context, 1, &es_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating ES program: %d\n", err);
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
            fprintf(stderr, "Error building ES program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_es", &err);
    update_kernel = clCreateKernel(program, "update_velocity_and_position", &err);
    levy_kernel = clCreateKernel(program, "levy_flight", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !update_kernel || !levy_kernel || !best_kernel) {
        fprintf(stderr, "Error creating ES kernels: %d\n", err);
        goto cleanup;
    }

    // Create buffers
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    population = (float*)malloc(population_size * dim * sizeof(float));
    fitness = (float*)malloc(population_size * sizeof(float));
    global_best = (float*)malloc((dim + 1) * sizeof(float)); // +1 for fitness
    best_idx_array = (int*)malloc(population_size * sizeof(int));
    temp_position = (double*)malloc(dim * sizeof(double));
    if (!bounds_float || !population || !fitness || !global_best || !best_idx_array || !temp_position) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < dim; i++) {
        global_best[i] = 0.0f;
    }
    global_best[dim] = INFINITY;

    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    velocity_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    local_best_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    local_best_cost_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    global_best_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (dim + 1) * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(int), NULL, &err);
    opt->population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    opt->fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !velocity_buffer || !local_best_buffer || 
        !local_best_cost_buffer || !global_best_buffer || !best_idx_buffer || 
        !opt->population_buffer || !opt->fitness_buffer) {
        fprintf(stderr, "Error creating ES buffers: %d\n", err);
        goto cleanup;
    }

    // Write initial data
    err = clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, global_best_buffer, CL_TRUE, 0, (dim + 1) * sizeof(float), global_best, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing initial buffers: %d\n", err);
        goto cleanup;
    }

    // Initialize population, velocities, and local bests
    uint seed = (uint)time(NULL);
    initialize_es(init_kernel, queue, opt->population_buffer, velocity_buffer, local_best_buffer, 
                  local_best_cost_buffer, global_best_buffer, bounds_buffer, dim, population_size, seed, &events[0]);

    // Evaluate initial population fitness on host and set initial global best
    err = clEnqueueReadBuffer(queue, opt->population_buffer, CL_TRUE, 0, population_size * dim * sizeof(float), population, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading population buffer: %d\n", err);
        goto cleanup;
    }
    float best_fitness = INFINITY;
    int best_idx = 0;
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            temp_position[d] = (double)population[i * dim + d];
        }
        fitness[i] = (float)objective_function(temp_position);
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            best_idx = i;
        }
    }
    err = clEnqueueWriteBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, local_best_cost_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing fitness buffers: %d\n", err);
        goto cleanup;
    }
    // Initialize global best
    opt->best_solution.fitness = (double)best_fitness;
    for (int d = 0; d < dim; d++) {
        opt->best_solution.position[d] = (double)population[best_idx * dim + d];
        global_best[d] = population[best_idx * dim + d];
    }
    global_best[dim] = best_fitness;
    err = clEnqueueWriteBuffer(queue, global_best_buffer, CL_TRUE, 0, (dim + 1) * sizeof(float), global_best, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing global best buffer: %d\n", err);
        goto cleanup;
    }

    // Main loop
    for (int iter = 0; iter < max_iter; iter++) {
        start_time = get_time_ms();

        // Decide between velocity update or Lévy flight
        if (((float)rand() / RAND_MAX) < LEVY_PROBABILITY) {
            levy_flight(levy_kernel, queue, opt->population_buffer, local_best_buffer, global_best_buffer, 
                        bounds_buffer, dim, population_size, seed, &events[1]);
        } else {
            update_velocity_and_position(update_kernel, queue, opt->population_buffer, velocity_buffer, 
                                        local_best_buffer, global_best_buffer, bounds_buffer, dim, 
                                        population_size, iter, max_iter, seed, &events[1]);
        }

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
            // Update local best if current fitness is better
            float local_best_cost;
            err = clEnqueueReadBuffer(queue, local_best_cost_buffer, CL_TRUE, i * sizeof(float), 
                                      sizeof(float), &local_best_cost, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading local best cost: %d\n", err);
                goto cleanup;
            }
            if (fitness[i] < local_best_cost) {
                err = clEnqueueWriteBuffer(queue, local_best_buffer, CL_TRUE, i * dim * sizeof(float), 
                                           dim * sizeof(float), &population[i * dim], 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error writing local best buffer: %d\n", err);
                    goto cleanup;
                }
                err = clEnqueueWriteBuffer(queue, local_best_cost_buffer, CL_TRUE, i * sizeof(float), 
                                           sizeof(float), &fitness[i], 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "Error writing local best cost buffer: %d\n", err);
                    goto cleanup;
                }
            }
        }
        err = clEnqueueWriteBuffer(queue, opt->fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing fitness buffer: %d\n", err);
            goto cleanup;
        }

        // Find best solution
        find_best(best_kernel, queue, opt->fitness_buffer, best_idx_buffer, population_size, &events[2]);

        // Update global best
        err = clEnqueueReadBuffer(queue, best_idx_buffer, CL_TRUE, 0, sizeof(int), &best_idx, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best_idx buffer: %d\n", err);
            goto cleanup;
        }
        best_fitness = fitness[best_idx];
        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)best_fitness;
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)population[best_idx * dim + d];
                global_best[d] = population[best_idx * dim + d];
            }
            global_best[dim] = best_fitness;
            err = clEnqueueWriteBuffer(queue, global_best_buffer, CL_TRUE, 0, (dim + 1) * sizeof(float), global_best, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing global best buffer: %d\n", err);
                goto cleanup;
            }
        }

        // Profiling
        cl_ulong time_start, time_end;
        double init_time = 0, update_time = 0, best_time = 0;
        for (int i = 0; i < 3; i++) {
            if (events[i]) {
                err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                    if (err == CL_SUCCESS) {
                        double time_ms = (time_end - time_start) / 1e6;
                        if (i == 0) init_time = time_ms;
                        else if (i == 1) update_time = time_ms;
                        else best_time = time_ms;
                    }
                }
            }
        }
        end_time = get_time_ms();
        printf("ES|%5d -----> %9.16f | Total: %.3f ms | Init: %.3f ms | Update/Levy: %.3f ms | Best: %.3f ms\n",
               iter + 1, opt->best_solution.fitness, end_time - start_time, 
               init_time, update_time, best_time);
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
    if (global_best) free(global_best);
    if (best_idx_array) free(best_idx_array);
    if (temp_position) free(temp_position);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (velocity_buffer) clReleaseMemObject(velocity_buffer);
    if (local_best_buffer) clReleaseMemObject(local_best_buffer);
    if (local_best_cost_buffer) clReleaseMemObject(local_best_cost_buffer);
    if (global_best_buffer) clReleaseMemObject(global_best_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (opt->population_buffer) clReleaseMemObject(opt->population_buffer);
    if (opt->fitness_buffer) clReleaseMemObject(opt->fitness_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (update_kernel) clReleaseKernel(update_kernel);
    if (levy_kernel) clReleaseKernel(levy_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 4; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (queue) clFinish(queue);
}
