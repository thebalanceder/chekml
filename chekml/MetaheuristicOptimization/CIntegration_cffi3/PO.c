/* PO.c - GPU-Optimized Political Optimizer with Flexible Objective Function */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
#include "generaloptimizer.h"
#include "PO.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Algorithm parameters
#define PARTIES 8
#define LAMBDA_RATE 1.0f

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
static const char* po_kernel_source =
"// Simple LCG random number generator\n"
"uint lcg_rand(uint* seed) {\n"
"    *seed = *seed * 1103515245 + 12345;\n"
"    return (*seed >> 16) & 0x7FFF;\n"
"}\n"
"float lcg_rand_float(uint* seed, float min, float max) {\n"
"    return min + (max - min) * ((float)lcg_rand(seed) / 0x7FFF);\n"
"}\n"
"// Clamp solution to bounds\n"
"void clamp(__global float* vec, const int dim, __local const float* bounds) {\n"
"    for (int i = 0; i < dim; i++) {\n"
"        vec[i] = max(bounds[2 * i], min(bounds[2 * i + 1], vec[i]));\n"
"    }\n"
"}\n"
"// Placeholder for objective function (user must implement)\n"
"float evaluate_fitness(__global const float* solution, const int dim) {\n"
"    // Example: Sphere function (sum of squares)\n"
"    float sum = 0.0f;\n"
"    for (int i = 0; i < dim; i++) {\n"
"        sum += solution[i] * solution[i];\n"
"    }\n"
"    return sum;\n"
"}\n"
"__kernel void initialize_population(\n"
"    __global float* population,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < pop_size) {\n"
"        uint local_seed = seed + id * dim;\n"
"        for (int d = 0; d < dim; d++) {\n"
"            float min = bounds[2 * d];\n"
"            float max = bounds[2 * d + 1];\n"
"            population[id * dim + d] = lcg_rand_float(&local_seed, min, max);\n"
"        }\n"
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
"        fitness[id] = evaluate_fitness(&population[id * dim], dim);\n"
"    }\n"
"}\n"
"__kernel void election_phase(\n"
"    __global const float* fitness,\n"
"    __global float* a_winners,\n"
"    __global int* a_winner_indices,\n"
"    const int areas,\n"
"    const int pop_size,\n"
"    __global const float* population,\n"
"    const int dim)\n"
"{\n"
"    int a = get_global_id(0);\n"
"    if (a < areas) {\n"
"        int start_idx = a;\n"
"        int step = areas;\n"
"        float min_fitness = INFINITY;\n"
"        int min_idx = -1;\n"
"        for (int i = start_idx; i < pop_size; i += step) {\n"
"            if (fitness[i] < min_fitness) {\n"
"                min_fitness = fitness[i];\n"
"                min_idx = i;\n"
"            }\n"
"        }\n"
"        a_winner_indices[a] = min_idx;\n"
"        for (int j = 0; j < dim; j++) {\n"
"            a_winners[a * dim + j] = population[min_idx * dim + j];\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void government_formation_phase(\n"
"    __global float* population,\n"
"    __global float* fitness,\n"
"    __global float* temp_pos,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int areas,\n"
"    const int pop_size,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int p = id / areas;\n"
"    int a = id % areas;\n"
"    if (id < pop_size) {\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        int party_start = p * areas;\n"
"        float min_fitness = INFINITY;\n"
"        int party_leader_idx = -1;\n"
"        for (int i = 0; i < areas; i++) {\n"
"            int idx = party_start + i;\n"
"            if (fitness[idx] < min_fitness) {\n"
"                min_fitness = fitness[idx];\n"
"                party_leader_idx = idx;\n"
"            }\n"
"        }\n"
"        int member_idx = party_start + a;\n"
"        if (member_idx != party_leader_idx) {\n"
"            uint local_seed = seed + id;\n"
"            float r = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"            for (int j = 0; j < dim; j++) {\n"
"                temp_pos[id * dim + j] = population[member_idx * dim + j] + r * (population[party_leader_idx * dim + j] - population[member_idx * dim + j]);\n"
"                temp_pos[id * dim + j] = max(local_bounds[2 * j], min(local_bounds[2 * j + 1], temp_pos[id * dim + j]));\n"
"            }\n"
"            float new_fitness = evaluate_fitness(&temp_pos[id * dim], dim);\n"
"            if (new_fitness < fitness[member_idx]) {\n"
"                for (int j = 0; j < dim; j++) {\n"
"                    population[member_idx * dim + j] = temp_pos[id * dim + j];\n"
"                }\n"
"                fitness[member_idx] = new_fitness;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void election_campaign_phase(\n"
"    __global float* population,\n"
"    __global float* fitness,\n"
"    __global const float* prev_positions,\n"
"    __global float* temp_pos,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int pop_size,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (id < pop_size) {\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        uint local_seed = seed + id;\n"
"        float r = lcg_rand_float(&local_seed, 0.0f, 1.0f);\n"
"        for (int j = 0; j < dim; j++) {\n"
"            temp_pos[id * dim + j] = population[id * dim + j] + r * (population[id * dim + j] - prev_positions[id * dim + j]);\n"
"            temp_pos[id * dim + j] = max(local_bounds[2 * j], min(local_bounds[2 * j + 1], temp_pos[id * dim + j]));\n"
"        }\n"
"        float new_fitness = evaluate_fitness(&temp_pos[id * dim], dim);\n"
"        if (new_fitness < fitness[id]) {\n"
"            for (int j = 0; j < dim; j++) {\n"
"                population[id * dim + j] = temp_pos[id * dim + j];\n"
"            }\n"
"            fitness[id] = new_fitness;\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void party_switching_phase(\n"
"    __global float* population,\n"
"    __global float* fitness,\n"
"    const int dim,\n"
"    const int areas,\n"
"    const int pop_size,\n"
"    const float psr,\n"
"    uint seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < pop_size) {\n"
"        int p = id / areas;\n"
"        int a = id % areas;\n"
"        uint local_seed = seed + id;\n"
"        if (lcg_rand_float(&local_seed, 0.0f, 1.0f) < psr) {\n"
"            int to_party = lcg_rand(&local_seed) % PARTIES;\n"
"            while (to_party == p) {\n"
"                to_party = lcg_rand(&local_seed) % PARTIES;\n"
"            }\n"
"            int to_start = to_party * areas;\n"
"            float max_fitness = -INFINITY;\n"
"            int to_least_fit_idx = -1;\n"
"            for (int i = to_start; i < to_start + areas; i++) {\n"
"                if (fitness[i] > max_fitness) {\n"
"                    max_fitness = fitness[i];\n"
"                    to_least_fit_idx = i;\n"
"                }\n"
"            }\n"
"            int from_idx = p * areas + a;\n"
"            for (int j = 0; j < dim; j++) {\n"
"                float temp = population[to_least_fit_idx * dim + j];\n"
"                population[to_least_fit_idx * dim + j] = population[from_idx * dim + j];\n"
"                population[from_idx * dim + j] = temp;\n"
"            }\n"
"            float temp_fitness = fitness[to_least_fit_idx];\n"
"            fitness[to_least_fit_idx] = fitness[from_idx];\n"
"            fitness[from_idx] = temp_fitness;\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void parliamentarism_phase(\n"
"    __global float* population,\n"
"    __global float* fitness,\n"
"    __global float* a_winners,\n"
"    __global const int* a_winner_indices,\n"
"    __global float* temp_pos,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int areas,\n"
"    uint seed,\n"
"    __local float* local_bounds)\n"
"{\n"
"    int a = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    if (a < areas) {\n"
"        if (local_id < dim * 2) {\n"
"            local_bounds[local_id] = bounds[local_id];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        uint local_seed = seed + a;\n"
"        for (int j = 0; j < dim; j++) {\n"
"            temp_pos[a * dim + j] = a_winners[a * dim + j];\n"
"        }\n"
"        int to_area = lcg_rand(&local_seed) % areas;\n"
"        while (to_area == a) {\n"
"            to_area = lcg_rand(&local_seed) % areas;\n"
"        }\n"
"        for (int j = 0; j < dim; j++) {\n"
"            float distance = fabs(a_winners[to_area * dim + j] - temp_pos[a * dim + j]);\n"
"            temp_pos[a * dim + j] = a_winners[to_area * dim + j] + (2.0f * lcg_rand_float(&local_seed, 0.0f, 1.0f) - 1.0f) * distance;\n"
"            temp_pos[a * dim + j] = max(local_bounds[2 * j], min(local_bounds[2 * j + 1], temp_pos[a * dim + j]));\n"
"        }\n"
"        float new_fitness = evaluate_fitness(&temp_pos[a * dim], dim);\n"
"        int winner_idx = a_winner_indices[a];\n"
"        if (new_fitness < fitness[winner_idx]) {\n"
"            for (int j = 0; j < dim; j++) {\n"
"                population[winner_idx * dim + j] = temp_pos[a * dim + j];\n"
"                a_winners[a * dim + j] = temp_pos[a * dim + j];\n"
"            }\n"
"            fitness[winner_idx] = new_fitness;\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void find_best(\n"
"    __global const float* fitness,\n"
"    __global int* best_idx,\n"
"    const int pop_size,\n"
"    __local float* local_fitness,\n"
"    __local int* local_indices)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    if (id < pop_size) {\n"
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

void PO_optimize(void *opt_void, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer *)opt_void;
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, eval_kernel = NULL, election_kernel = NULL, gov_formation_kernel = NULL, campaign_kernel = NULL, party_switching_kernel = NULL, parliamentarism_kernel = NULL, best_kernel = NULL;
    cl_mem bounds_buffer = NULL, population_buffer = NULL, fitness_buffer = NULL, prev_positions_buffer = NULL, a_winners_buffer = NULL, a_winner_indices_buffer = NULL, temp_pos_buffer = NULL, best_idx_buffer = NULL;
    float *bounds_float = NULL, *population = NULL, *fitness = NULL, *prev_positions = NULL, *a_winners = NULL, *temp_pos = NULL;
    int *a_winner_indices = NULL, *best_idx_array = NULL;
    cl_event events[6] = {0};
    double start_time, end_time;
    cl_context context = NULL;
    cl_command_queue queue = NULL;

    // Validate input
    if (!opt || !opt->bounds || !opt->population || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        return;
    }
    const int dim = opt->dim;
    const int pop_size = 256; // Fixed for GPU scalability, divisible by PARTIES
    const int max_iter = 1000;
    const int areas = pop_size / PARTIES;
    if (dim < 1 || dim > 32) {
        fprintf(stderr, "Error: Invalid dim (%d), must be between 1 and 32\n", dim);
        return;
    }
    if (pop_size % PARTIES != 0) {
        fprintf(stderr, "Error: pop_size (%d) must be divisible by PARTIES (%d)\n", pop_size, PARTIES);
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
        platform = platforms[i];
        device = devices[0]; // Use first available GPU
        free(devices);
        break;
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
    program = clCreateProgramWithSource(context, 1, &po_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating PO program: %d\n", err);
        goto cleanup;
    }
    char build_options[] = "-D PARTIES=8";
    err = clBuildProgram(program, 1, &device, build_options, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building PO program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        program = NULL;
        goto cleanup;
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    eval_kernel = clCreateKernel(program, "evaluate_population", &err);
    election_kernel = clCreateKernel(program, "election_phase", &err);
    gov_formation_kernel = clCreateKernel(program, "government_formation_phase", &err);
    campaign_kernel = clCreateKernel(program, "election_campaign_phase", &err);
    party_switching_kernel = clCreateKernel(program, "party_switching_phase", &err);
    parliamentarism_kernel = clCreateKernel(program, "parliamentarism_phase", &err);
    best_kernel = clCreateKernel(program, "find_best", &err);
    if (err != CL_SUCCESS || !init_kernel || !eval_kernel || !election_kernel || !gov_formation_kernel || !campaign_kernel || !party_switching_kernel || !parliamentarism_kernel || !best_kernel) {
        fprintf(stderr, "Error creating PO kernels: %d\n", err);
        goto cleanup;
    }

    // Allocate host memory
    bounds_float = (float*)malloc(2 * dim * sizeof(float));
    population = (float*)malloc(pop_size * dim * sizeof(float));
    fitness = (float*)malloc(pop_size * sizeof(float));
    prev_positions = (float*)malloc(pop_size * dim * sizeof(float));
    a_winners = (float*)malloc(areas * dim * sizeof(float));
    a_winner_indices = (int*)malloc(areas * sizeof(int));
    temp_pos = (float*)malloc(pop_size * dim * sizeof(float));
    best_idx_array = (int*)malloc(pop_size * sizeof(int));
    if (!bounds_float || !population || !fitness || !prev_positions || !a_winners || !a_winner_indices || !temp_pos || !best_idx_array) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup;
    }
    for (int i = 0; i < 2 * dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }

    // Create buffers
    bounds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    population_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    fitness_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * sizeof(float), NULL, &err);
    prev_positions_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    a_winners_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, areas * dim * sizeof(float), NULL, &err);
    a_winner_indices_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, areas * sizeof(int), NULL, &err);
    temp_pos_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(float), NULL, &err);
    best_idx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pop_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer || !population_buffer || !fitness_buffer || !prev_positions_buffer || 
        !a_winners_buffer || !a_winner_indices_buffer || !temp_pos_buffer || !best_idx_buffer) {
        fprintf(stderr, "Error creating PO buffers: %d\n", err);
        goto cleanup;
    }

    // Write bounds
    err = clEnqueueWriteBuffer(queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bounds buffer: %d\n", err);
        goto cleanup;
    }

    uint seed = (uint)time(NULL);

    // Initialize population
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(init_kernel, 3, sizeof(int), &pop_size);
    err |= clSetKernelArg(init_kernel, 4, sizeof(uint), &seed);
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

    // Evaluate initial population
    err = clSetKernelArg(eval_kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(eval_kernel, 1, sizeof(cl_mem), &fitness_buffer);
    err |= clSetKernelArg(eval_kernel, 2, sizeof(int), &dim);
    err |= clSetKernelArg(eval_kernel, 3, sizeof(int), &pop_size);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting eval kernel args: %d\n", err);
        goto cleanup;
    }
    err = clEnqueueNDRangeKernel(queue, eval_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[1]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing eval kernel: %d\n", err);
        goto cleanup;
    }

    // Copy population to prev_positions
    err = clEnqueueCopyBuffer(queue, population_buffer, prev_positions_buffer, 0, 0, pop_size * dim * sizeof(float), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error copying population to prev_positions: %d\n", err);
        goto cleanup;
    }

    // Initial phases
    err = clSetKernelArg(election_kernel, 0, sizeof(cl_mem), &fitness_buffer);
    err |= clSetKernelArg(election_kernel, 1, sizeof(cl_mem), &a_winners_buffer);
    err |= clSetKernelArg(election_kernel, 2, sizeof(cl_mem), &a_winner_indices_buffer);
    err |= clSetKernelArg(election_kernel, 3, sizeof(int), &areas);
    err |= clSetKernelArg(election_kernel, 4, sizeof(int), &pop_size);
    err |= clSetKernelArg(election_kernel, 5, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(election_kernel, 6, sizeof(int), &dim);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting election kernel args: %d\n", err);
        goto cleanup;
    }
    global_work_size = ((areas + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, election_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[2]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing election kernel: %d\n", err);
        goto cleanup;
    }

    err = clSetKernelArg(gov_formation_kernel, 0, sizeof(cl_mem), &population_buffer);
    err |= clSetKernelArg(gov_formation_kernel, 1, sizeof(cl_mem), &fitness_buffer);
    err |= clSetKernelArg(gov_formation_kernel, 2, sizeof(cl_mem), &temp_pos_buffer);
    err |= clSetKernelArg(gov_formation_kernel, 3, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(gov_formation_kernel, 4, sizeof(int), &dim);
    err |= clSetKernelArg(gov_formation_kernel, 5, sizeof(int), &areas);
    err |= clSetKernelArg(gov_formation_kernel, 6, sizeof(int), &pop_size);
    err |= clSetKernelArg(gov_formation_kernel, 7, sizeof(uint), &seed);
    err |= clSetKernelArg(gov_formation_kernel, 8, 2 * dim * sizeof(float), NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting gov formation kernel args: %d\n", err);
        goto cleanup;
    }
    global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, gov_formation_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[3]);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing gov formation kernel: %d\n", err);
        goto cleanup;
    }

    // Main PO loop
    for (int t = 0; t < max_iter; t++) {
        start_time = get_time_ms();

        // Update prev_positions
        err = clEnqueueCopyBuffer(queue, population_buffer, prev_positions_buffer, 0, 0, pop_size * dim * sizeof(float), 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error copying population to prev_positions: %d\n", err);
            goto cleanup;
        }

        // Election campaign phase
        err = clSetKernelArg(campaign_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(campaign_kernel, 1, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(campaign_kernel, 2, sizeof(cl_mem), &prev_positions_buffer);
        err |= clSetKernelArg(campaign_kernel, 3, sizeof(cl_mem), &temp_pos_buffer);
        err |= clSetKernelArg(campaign_kernel, 4, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(campaign_kernel, 5, sizeof(int), &dim);
        err |= clSetKernelArg(campaign_kernel, 6, sizeof(int), &pop_size);
        err |= clSetKernelArg(campaign_kernel, 7, sizeof(uint), &seed);
        err |= clSetKernelArg(campaign_kernel, 8, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting campaign kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, campaign_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[4]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing campaign kernel: %d\n", err);
            goto cleanup;
        }

        // Party switching phase
        float psr = (1.0f - t * (1.0f / max_iter)) * LAMBDA_RATE;
        err = clSetKernelArg(party_switching_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(party_switching_kernel, 1, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(party_switching_kernel, 2, sizeof(int), &dim);
        err |= clSetKernelArg(party_switching_kernel, 3, sizeof(int), &areas);
        err |= clSetKernelArg(party_switching_kernel, 4, sizeof(int), &pop_size);
        err |= clSetKernelArg(party_switching_kernel, 5, sizeof(float), &psr);
        err |= clSetKernelArg(party_switching_kernel, 6, sizeof(uint), &seed);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting party switching kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, party_switching_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &events[5]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing party switching kernel: %d\n", err);
            goto cleanup;
        }

        // Election phase
        err = clSetKernelArg(election_kernel, 0, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(election_kernel, 1, sizeof(cl_mem), &a_winners_buffer);
        err |= clSetKernelArg(election_kernel, 2, sizeof(cl_mem), &a_winner_indices_buffer);
        err |= clSetKernelArg(election_kernel, 3, sizeof(int), &areas);
        err |= clSetKernelArg(election_kernel, 4, sizeof(int), &pop_size);
        err |= clSetKernelArg(election_kernel, 5, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(election_kernel, 6, sizeof(int), &dim);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting election kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((areas + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, election_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing election kernel: %d\n", err);
            goto cleanup;
        }

        // Government formation phase
        err = clSetKernelArg(gov_formation_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(gov_formation_kernel, 1, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(gov_formation_kernel, 2, sizeof(cl_mem), &temp_pos_buffer);
        err |= clSetKernelArg(gov_formation_kernel, 3, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(gov_formation_kernel, 4, sizeof(int), &dim);
        err |= clSetKernelArg(gov_formation_kernel, 5, sizeof(int), &areas);
        err |= clSetKernelArg(gov_formation_kernel, 6, sizeof(int), &pop_size);
        err |= clSetKernelArg(gov_formation_kernel, 7, sizeof(uint), &seed);
        err |= clSetKernelArg(gov_formation_kernel, 8, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting gov formation kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((pop_size + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, gov_formation_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing gov formation kernel: %d\n", err);
            goto cleanup;
        }

        // Parliamentarism phase
        err = clSetKernelArg(parliamentarism_kernel, 0, sizeof(cl_mem), &population_buffer);
        err |= clSetKernelArg(parliamentarism_kernel, 1, sizeof(cl_mem), &fitness_buffer);
        err |= clSetKernelArg(parliamentarism_kernel, 2, sizeof(cl_mem), &a_winners_buffer);
        err |= clSetKernelArg(parliamentarism_kernel, 3, sizeof(cl_mem), &a_winner_indices_buffer);
        err |= clSetKernelArg(parliamentarism_kernel, 4, sizeof(cl_mem), &temp_pos_buffer);
        err |= clSetKernelArg(parliamentarism_kernel, 5, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(parliamentarism_kernel, 6, sizeof(int), &dim);
        err |= clSetKernelArg(parliamentarism_kernel, 7, sizeof(int), &areas);
        err |= clSetKernelArg(parliamentarism_kernel, 8, sizeof(uint), &seed);
        err |= clSetKernelArg(parliamentarism_kernel, 9, 2 * dim * sizeof(float), NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting parliamentarism kernel args: %d\n", err);
            goto cleanup;
        }
        global_work_size = ((areas + local_work_size - 1) / local_work_size) * local_work_size;
        err = clEnqueueNDRangeKernel(queue, parliamentarism_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing parliamentarism kernel: %d\n", err);
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
        err = clEnqueueNDRangeKernel(queue, best_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing best kernel: %d\n", err);
            goto cleanup;
        }

        // Update best solution
        int best_idx;
        float best_fitness;
        err = clEnqueueReadBuffer(queue, best_idx_buffer, CL_TRUE, 0, sizeof(int), &best_idx, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best_idx buffer: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, best_idx * sizeof(float), sizeof(float), &best_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading best fitness: %d\n", err);
            goto cleanup;
        }
        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = (double)best_fitness;
            err = clEnqueueReadBuffer(queue, population_buffer, CL_TRUE, best_idx * dim * sizeof(float), dim * sizeof(float), population, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best solution: %d\n", err);
                goto cleanup;
            }
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = (double)population[d];
            }
        }

        // Profiling
        cl_ulong time_start, time_end;
        double init_time = 0, eval_time = 0, election_time = 0, gov_formation_time = 0, campaign_time = 0, party_switching_time = 0;
        cl_ulong queue_properties;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(cl_ulong), &queue_properties, NULL);
        if (err == CL_SUCCESS && (queue_properties & CL_QUEUE_PROFILING_ENABLE)) {
            for (int i = 0; i < 6; i++) {
                if (events[i]) {
                    err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
                    if (err == CL_SUCCESS) {
                        err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
                        if (err == CL_SUCCESS) {
                            double time_ms = (time_end - time_start) / 1e6;
                            if (i == 0) init_time = time_ms;
                            else if (i == 1) eval_time = time_ms;
                            else if (i == 2) election_time = time_ms;
                            else if (i == 3) gov_formation_time = time_ms;
                            else if (i == 4) campaign_time = time_ms;
                            else if (i == 5) party_switching_time = time_ms;
                        }
                    }
                }
            }
            end_time = get_time_ms();
            printf("PO | Iteration %4d -> Best Fitness = %.16f | Total: %.3f ms | Init: %.3f ms | Eval: %.3f ms | Election: %.3f ms | GovFormation: %.3f ms | Campaign: %.3f ms | PartySwitching: %.3f ms\n",
                   t + 1, opt->best_solution.fitness, end_time - start_time, init_time, eval_time, election_time, gov_formation_time, campaign_time, party_switching_time);
        } else {
            end_time = get_time_ms();
            printf("PO | Iteration %4d -> Best Fitness = %.16f | Total: %.3f ms | Profiling disabled\n",
                   t + 1, opt->best_solution.fitness, end_time - start_time);
        }
    }

cleanup:
    if (bounds_float) free(bounds_float);
    if (population) free(population);
    if (fitness) free(fitness);
    if (prev_positions) free(prev_positions);
    if (a_winners) free(a_winners);
    if (a_winner_indices) free(a_winner_indices);
    if (temp_pos) free(temp_pos);
    if (best_idx_array) free(best_idx_array);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
    if (population_buffer) clReleaseMemObject(population_buffer);
    if (fitness_buffer) clReleaseMemObject(fitness_buffer);
    if (prev_positions_buffer) clReleaseMemObject(prev_positions_buffer);
    if (a_winners_buffer) clReleaseMemObject(a_winners_buffer);
    if (a_winner_indices_buffer) clReleaseMemObject(a_winner_indices_buffer);
    if (temp_pos_buffer) clReleaseMemObject(temp_pos_buffer);
    if (best_idx_buffer) clReleaseMemObject(best_idx_buffer);
    if (init_kernel) clReleaseKernel(init_kernel);
    if (eval_kernel) clReleaseKernel(eval_kernel);
    if (election_kernel) clReleaseKernel(election_kernel);
    if (gov_formation_kernel) clReleaseKernel(gov_formation_kernel);
    if (campaign_kernel) clReleaseKernel(campaign_kernel);
    if (party_switching_kernel) clReleaseKernel(party_switching_kernel);
    if (parliamentarism_kernel) clReleaseKernel(parliamentarism_kernel);
    if (best_kernel) clReleaseKernel(best_kernel);
    if (program) clReleaseProgram(program);
    for (int i = 0; i < 6; i++) {
        if (events[i]) clReleaseEvent(events[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (queue) clFinish(queue);
}
