#include "CA.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Static alpha variable
static float alpha = 0.5f;

// OpenCL kernel source for CA
const char* ca_kernel_source =
    "// Linear congruential generator for random floats\n"
    "float rand_float(uint* seed) {\n"
    "    *seed = *seed * 1103515245 + 12345;\n"
    "    return (float)(*seed & 0x7fffffff) / 2147483647.0f;\n"
    "}\n"
    "\n"
    "// Atmospheric absorption coefficient\n"
    "float CA_CoefCalculate(float F, float T) {\n"
    "    float pres = 1.0f;\n"
    "    float relh = 50.0f;\n"
    "    float freq_hum = F;\n"
    "    float temp = T + 273.0f;\n"
    "    if (temp < 1e-10f) temp = 1e-10f;\n"
    "    float C_humid = 4.6151f - 6.8346f * pow(273.15f / temp, 1.261f);\n"
    "    float hum = relh * pow(10.0f, C_humid) * pres;\n"
    "    float tempr = temp / 293.15f;\n"
    "    float frO = pres * (24.0f + 4.04e4f * hum * (0.02f + hum) / (0.391f + hum));\n"
    "    float frN = pres * pow(tempr, -0.5f) * (9.0f + 280.0f * hum * exp(-4.17f * (pow(tempr, -1.0f/3.0f) - 1.0f)));\n"
    "    float freq_hum_sq = freq_hum * freq_hum;\n"
    "    float alpha = 8.686f * freq_hum_sq * (\n"
    "        1.84e-11f * (1.0f / pres) * sqrt(tempr) +\n"
    "        pow(tempr, -2.5f) * (\n"
    "            0.01275f * (exp(-2239.1f / temp) / (frO + freq_hum_sq / frO)) +\n"
    "            0.1068f * (exp(-3352.0f / temp) / (frN + freq_hum_sq / frN))\n"
    "        )\n"
    "    );\n"
    "    return round(alpha * 1000.0f) / 1000.0f;\n"
    "}\n"
    "\n"
    "// Kernel for cricket update phase\n"
    "__kernel void CA_cricket_update(__global float* population, __global float* best_position,\n"
    "                               __global float* bounds, __global float* Q, __global float* v,\n"
    "                               __global float* S, __global float* M, __global float* F,\n"
    "                               __global float* T, __global float* C, __global float* V,\n"
    "                               __global float* Z, __global float* scale, int dim,\n"
    "                               int population_size, float alpha, uint seed) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= population_size) return;\n"
    "    uint local_seed = seed + i;\n"
    "\n"
    "    float F_mean = 0.0f, SumT = 0.0f;\n"
    "    for (int j = 0; j < dim; j++) {\n"
    "        float N = (float)(rand_float(&local_seed) * 121.0f);\n"
    "        T[j] = 0.891797f * N + 40.0252f;\n"
    "        T[j] = (T[j] < 55.0f) ? 55.0f : (T[j] > 180.0f) ? 180.0f : T[j];\n"
    "        C[j] = (5.0f / 9.0f) * (T[j] - 32.0f);\n"
    "        SumT += C[j];\n"
    "        V[j] = 20.1f * sqrt(273.0f + C[j]);\n"
    "        V[j] = sqrt(V[j]) / 1000.0f;\n"
    "        Z[j] = population[i * dim + j] - best_position[j];\n"
    "        F[j] = (fabs(Z[j]) > 1e-10f) ? V[j] / Z[j] : 0.0f;\n"
    "        F_mean += F[j];\n"
    "    }\n"
    "    F_mean /= dim;\n"
    "    SumT /= dim;\n"
    "\n"
    "    Q[i] = 0.0f + (F_mean - 0.0f) * rand_float(&local_seed);\n"
    "\n"
    "    for (int j = 0; j < dim; j++) {\n"
    "        v[i * dim + j] += (population[i * dim + j] - best_position[j]) * Q[i] + V[j];\n"
    "        S[i * dim + j] = population[i * dim + j] + v[i * dim + j];\n"
    "    }\n"
    "\n"
    "    float SumF = F_mean + 10000.0f;\n"
    "    float gamma = CA_CoefCalculate(SumF, SumT);\n"
    "\n"
    "    for (int j = 0; j < dim; j++) {\n"
    "        float tmpf = alpha * (rand_float(&local_seed) - 0.5f) * scale[j];\n"
    "        M[i * dim + j] = population[i * dim + j] * (1.0f - 0.2f) + best_position[j] * 0.2f + tmpf;\n"
    "    }\n"
    "\n"
    "    float r = rand_float(&local_seed);\n"
    "    for (int j = 0; j < dim; j++) {\n"
    "        population[i * dim + j] = (r > gamma) ? S[i * dim + j] : M[i * dim + j];\n"
    "    }\n"
    "}\n"
    "\n"
    "// Kernel for fitness-based updates\n"
    "__kernel void CA_fitness_update(__global float* population, __global float* fitness,\n"
    "                               __global float* best_position, __global float* scale,\n"
    "                               int dim, int population_size, float alpha, uint seed) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= population_size) return;\n"
    "    uint local_seed = seed + i;\n"
    "\n"
    "    for (int k = 0; k < population_size; k++) {\n"
    "        if (fitness[i] < fitness[k]) {\n"
    "            float distance = 0.0f;\n"
    "            for (int j = 0; j < dim; j++) {\n"
    "                float diff = population[i * dim + j] - population[k * dim + j];\n"
    "                distance += diff * diff;\n"
    "            }\n"
    "            distance = sqrt(distance);\n"
    "            float distance_sq = distance * distance;\n"
    "            float PS = fitness[i] * (4.0f * 3.141592653589793f * distance_sq);\n"
    "            float Lp = PS + 10.0f * log10(1.0f / (4.0f * 3.141592653589793f * distance_sq));\n"
    "            float Aatm = (7.4f * (10000.0f * 10000.0f * distance) / (50.0f * 1e-8f));\n"
    "            float RLP = Lp - Aatm;\n"
    "            float K = RLP * exp(-0.001f * distance_sq);\n"
    "            float beta = K + 0.2f;\n"
    "\n"
    "            for (int j = 0; j < dim; j++) {\n"
    "                float tmpf = alpha * (rand_float(&local_seed) - 0.5f) * scale[j];\n"
    "                population[i * dim + j] = population[i * dim + j] * (1.0f - beta) +\n"
    "                                          population[k * dim + j] * beta + tmpf;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "}\n";

// Helper function to release OpenCL resources safely
static void safe_release_cl_resources(cl_program program, cl_kernel update_kernel, cl_kernel fitness_kernel,
                                      cl_mem Q_buffer, cl_mem v_buffer, cl_mem S_buffer, cl_mem M_buffer,
                                      cl_mem F_buffer, cl_mem T_buffer, cl_mem C_buffer, cl_mem V_buffer,
                                      cl_mem Z_buffer, cl_mem scale_buffer, cl_mem bounds_buffer) {
    if (program) clReleaseProgram(program);
    if (update_kernel) clReleaseKernel(update_kernel);
    if (fitness_kernel) clReleaseKernel(fitness_kernel);
    if (Q_buffer) clReleaseMemObject(Q_buffer);
    if (v_buffer) clReleaseMemObject(v_buffer);
    if (S_buffer) clReleaseMemObject(S_buffer);
    if (M_buffer) clReleaseMemObject(M_buffer);
    if (F_buffer) clReleaseMemObject(F_buffer);
    if (T_buffer) clReleaseMemObject(T_buffer);
    if (C_buffer) clReleaseMemObject(C_buffer);
    if (V_buffer) clReleaseMemObject(V_buffer);
    if (Z_buffer) clReleaseMemObject(Z_buffer);
    if (scale_buffer) clReleaseMemObject(scale_buffer);
    if (bounds_buffer) clReleaseMemObject(bounds_buffer);
}

// Main Optimization Function
void CA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel update_kernel = NULL, fitness_kernel = NULL;
    cl_mem Q_buffer = NULL, v_buffer = NULL, S_buffer = NULL, M_buffer = NULL;
    cl_mem F_buffer = NULL, T_buffer = NULL, C_buffer = NULL, V_buffer = NULL;
    cl_mem Z_buffer = NULL, scale_buffer = NULL, bounds_buffer = NULL;
    float *population_float = NULL, *fitness_float = NULL, *scale = NULL, *bounds = NULL;

    // Validate input
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(EXIT_FAILURE);
    }

    const int dim = opt->dim;
    const int population_size = opt->population_size;
    const int max_iter = opt->max_iter;
    if (dim < 1 || population_size < 1 || max_iter < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), or max_iter (%d)\n", 
                dim, population_size, max_iter);
        exit(EXIT_FAILURE);
    }
    if (dim > 32) {
        fprintf(stderr, "Error: dim (%d) exceeds maximum supported value of 32\n", dim);
        exit(EXIT_FAILURE);
    }

    // Query device capabilities
    size_t max_work_group_size;
    err = clGetDeviceInfo(opt->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying max work group size: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = max_work_group_size < 64 ? max_work_group_size : 64;

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = 0.0;
    }

    // Create OpenCL program
    program = clCreateProgramWithSource(opt->context, 1, &ca_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating CA program: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building CA program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    update_kernel = clCreateKernel(program, "CA_cricket_update", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating update_kernel: %d\n", err);
        goto cleanup;
    }
    fitness_kernel = clCreateKernel(program, "CA_fitness_update", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating fitness_kernel: %d\n", err);
        goto cleanup;
    }

    // Create additional buffers
    Q_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating Q_buffer: %d\n", err); goto cleanup; }
    v_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating v_buffer: %d\n", err); goto cleanup; }
    S_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating S_buffer: %d\n", err); goto cleanup; }
    M_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating M_buffer: %d\n", err); goto cleanup; }
    F_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating F_buffer: %d\n", err); goto cleanup; }
    T_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating T_buffer: %d\n", err); goto cleanup; }
    C_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating C_buffer: %d\n", err); goto cleanup; }
    V_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating V_buffer: %d\n", err); goto cleanup; }
    Z_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating Z_buffer: %d\n", err); goto cleanup; }
    scale_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating scale_buffer: %d\n", err); goto cleanup; }
    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating bounds_buffer: %d\n", err); goto cleanup; }

    // Initialize scale and bounds
    scale = (float *)malloc(dim * sizeof(float));
    if (!scale) { fprintf(stderr, "Error allocating scale\n"); goto cleanup; }
    bounds = (float *)malloc(2 * dim * sizeof(float));
    if (!bounds) { fprintf(stderr, "Error allocating bounds\n"); goto cleanup; }
    for (int j = 0; j < dim; j++) {
        scale[j] = (float)(opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        bounds[2 * j] = (float)opt->bounds[2 * j];
        bounds[2 * j + 1] = (float)opt->bounds[2 * j + 1];
    }
    err = clEnqueueWriteBuffer(opt->queue, scale_buffer, CL_TRUE, 0, dim * sizeof(float), scale, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error writing scale_buffer: %d\n", err); goto cleanup; }
    err = clEnqueueWriteBuffer(opt->queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error writing bounds_buffer: %d\n", err); goto cleanup; }

    // Set kernel arguments for update_kernel
    err = clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &opt->population_buffer); // best_position uses same buffer for simplicity
    err |= clSetKernelArg(update_kernel, 2, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(update_kernel, 3, sizeof(cl_mem), &Q_buffer);
    err |= clSetKernelArg(update_kernel, 4, sizeof(cl_mem), &v_buffer);
    err |= clSetKernelArg(update_kernel, 5, sizeof(cl_mem), &S_buffer);
    err |= clSetKernelArg(update_kernel, 6, sizeof(cl_mem), &M_buffer);
    err |= clSetKernelArg(update_kernel, 7, sizeof(cl_mem), &F_buffer);
    err |= clSetKernelArg(update_kernel, 8, sizeof(cl_mem), &T_buffer);
    err |= clSetKernelArg(update_kernel, 9, sizeof(cl_mem), &C_buffer);
    err |= clSetKernelArg(update_kernel, 10, sizeof(cl_mem), &V_buffer);
    err |= clSetKernelArg(update_kernel, 11, sizeof(cl_mem), &Z_buffer);
    err |= clSetKernelArg(update_kernel, 12, sizeof(cl_mem), &scale_buffer);
    err |= clSetKernelArg(update_kernel, 13, sizeof(int), &dim);
    err |= clSetKernelArg(update_kernel, 14, sizeof(int), &population_size);
    err |= clSetKernelArg(update_kernel, 15, sizeof(float), &alpha);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error setting update_kernel args: %d\n", err); goto cleanup; }

    // Set kernel arguments for fitness_kernel
    err = clSetKernelArg(fitness_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(fitness_kernel, 1, sizeof(cl_mem), &opt->fitness_buffer);
    err |= clSetKernelArg(fitness_kernel, 2, sizeof(cl_mem), &opt->population_buffer); // best_position
    err |= clSetKernelArg(fitness_kernel, 3, sizeof(cl_mem), &scale_buffer);
    err |= clSetKernelArg(fitness_kernel, 4, sizeof(int), &dim);
    err |= clSetKernelArg(fitness_kernel, 5, sizeof(int), &population_size);
    err |= clSetKernelArg(fitness_kernel, 6, sizeof(float), &alpha);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error setting fitness_kernel args: %d\n", err); goto cleanup; }

    // Allocate host buffers
    population_float = (float *)malloc(dim * population_size * sizeof(float));
    if (!population_float) { fprintf(stderr, "Error allocating population_float\n"); goto cleanup; }
    fitness_float = (float *)malloc(population_size * sizeof(float));
    if (!fitness_float) { fprintf(stderr, "Error allocating fitness_float\n"); goto cleanup; }

    // Verify population memory
    for (int i = 0; i < population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: opt->population[%d].position is NULL\n", i);
            goto cleanup;
        }
    }
    if (!opt->best_solution.position) {
        fprintf(stderr, "Error: opt->best_solution.position is NULL\n");
        goto cleanup;
    }

    // Main optimization loop
    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    int iter = 0;
    while (opt->best_solution.fitness > CA_TOL && iter < max_iter) {
        // Update alpha
        alpha *= 0.97f;
        unsigned int seed = (unsigned int)time(NULL) + iter;

        // Run cricket update kernel
        err = clSetKernelArg(update_kernel, 16, sizeof(unsigned int), &seed);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error setting update_kernel seed: %d\n", err); goto cleanup; }
        err = clEnqueueNDRangeKernel(opt->queue, update_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error enqueuing update_kernel: %d\n", err); goto cleanup; }

        // Read population for fitness evaluation
        err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, dim * population_size * sizeof(float), population_float, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error reading population_buffer: %d\n", err); goto cleanup; }
        for (int i = 0; i < population_size; i++) {
            double position[32]; // Assumes dim <= 32
            for (int j = 0; j < dim; j++) {
                position[j] = (double)population_float[i * dim + j];
            }
            double new_fitness = objective_function(position);
            fitness_float[i] = (float)new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < dim; j++) {
                    opt->best_solution.position[j] = position[j];
                }
            }
        }
        err = clEnqueueWriteBuffer(opt->queue, opt->fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness_float, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error writing fitness_buffer: %d\n", err); goto cleanup; }

        // Run fitness update kernel
        err = clSetKernelArg(fitness_kernel, 7, sizeof(unsigned int), &seed);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error setting fitness_kernel seed: %d\n", err); goto cleanup; }
        err = clEnqueueNDRangeKernel(opt->queue, fitness_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error enqueuing fitness_kernel: %d\n", err); goto cleanup; }

        // Enforce bounds
        err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, dim * population_size * sizeof(float), population_float, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error reading population_buffer for bounds: %d\n", err); goto cleanup; }
        for (int i = 0; i < population_size; i++) {
            for (int j = 0; j < dim; j++) {
                opt->population[i].position[j] = (double)population_float[i * dim + j];
                if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                    opt->population[i].position[j] = opt->bounds[2 * j];
                } else if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                    opt->population[i].position[j] = opt->bounds[2 * j + 1];
                }
            }
        }
        for (int i = 0; i < population_size; i++) {
            for (int j = 0; j < dim; j++) {
                population_float[i * dim + j] = (float)opt->population[i].position[j];
            }
        }
        err = clEnqueueWriteBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, dim * population_size * sizeof(float), population_float, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error writing population_buffer: %d\n", err); goto cleanup; }

        iter++;
        printf("Iteration %d: Best Fitness = %f\n", iter, opt->best_solution.fitness);
    }

    // Final population read
    err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, dim * population_size * sizeof(float), population_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error reading final population_buffer: %d\n", err); goto cleanup; }

    // Debug print before cleanup
    fprintf(stderr, "Finished optimization loop, starting cleanup\n");

cleanup:
    if (population_float) free(population_float);
    if (fitness_float) free(fitness_float);
    if (scale) free(scale);
    if (bounds) free(bounds);
    population_float = NULL;
    fitness_float = NULL;
    scale = NULL;
    bounds = NULL;

    safe_release_cl_resources(program, update_kernel, fitness_kernel, Q_buffer, v_buffer, S_buffer, M_buffer,
                             F_buffer, T_buffer, C_buffer, V_buffer, Z_buffer, scale_buffer, bounds_buffer);
    program = NULL;
    update_kernel = NULL;
    fitness_kernel = NULL;
    Q_buffer = NULL;
    v_buffer = NULL;
    S_buffer = NULL;
    M_buffer = NULL;
    F_buffer = NULL;
    T_buffer = NULL;
    C_buffer = NULL;
    V_buffer = NULL;
    Z_buffer = NULL;
    scale_buffer = NULL;
    bounds_buffer = NULL;

    clFinish(opt->queue);
    fprintf(stderr, "Cleanup completed\n");
}

// Optimization with History for Benchmarking
void CA_optimize_with_history(Optimizer *opt, double (*objective_function)(double *), double **history, int *history_size, int max_history) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel update_kernel = NULL, fitness_kernel = NULL;
    cl_mem Q_buffer = NULL, v_buffer = NULL, S_buffer = NULL, M_buffer = NULL;
    cl_mem F_buffer = NULL, T_buffer = NULL, C_buffer = NULL, V_buffer = NULL;
    cl_mem Z_buffer = NULL, scale_buffer = NULL, bounds_buffer = NULL;
    float *population_float = NULL, *fitness_float = NULL, *scale = NULL, *bounds = NULL;

    // Validate input
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position || !objective_function || !history) {
        fprintf(stderr, "Error: Invalid Optimizer structure, null pointers, or history\n");
        exit(EXIT_FAILURE);
    }

    const int dim = opt->dim;
    const int population_size = opt->population_size;
    const int max_iter = opt->max_iter;
    if (dim < 1 || population_size < 1 || max_iter < 1 || max_history < 1) {
        fprintf(stderr, "Error: Invalid dim (%d), population_size (%d), max_iter (%d), or max_history (%d)\n", 
                dim, population_size, max_iter, max_history);
        exit(EXIT_FAILURE);
    }
    if (dim > 32) {
        fprintf(stderr, "Error: dim (%d) exceeds maximum supported value of 32\n", dim);
        exit(EXIT_FAILURE);
    }

    // Query device capabilities
    size_t max_work_group_size;
    err = clGetDeviceInfo(opt->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error querying max work group size: %d\n", err);
        exit(EXIT_FAILURE);
    }
    size_t local_work_size = max_work_group_size < 64 ? max_work_group_size : 64;

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = 0.0;
    }

    // Create OpenCL program
    program = clCreateProgramWithSource(opt->context, 1, &ca_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating CA program: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building CA program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    update_kernel = clCreateKernel(program, "CA_cricket_update", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating update_kernel: %d\n", err);
        goto cleanup;
    }
    fitness_kernel = clCreateKernel(program, "CA_fitness_update", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating fitness_kernel: %d\n", err);
        goto cleanup;
    }

    // Create additional buffers
    Q_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating Q_buffer: %d\n", err); goto cleanup; }
    v_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating v_buffer: %d\n", err); goto cleanup; }
    S_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating S_buffer: %d\n", err); goto cleanup; }
    M_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating M_buffer: %d\n", err); goto cleanup; }
    F_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating F_buffer: %d\n", err); goto cleanup; }
    T_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating T_buffer: %d\n", err); goto cleanup; }
    C_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating C_buffer: %d\n", err); goto cleanup; }
    V_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating V_buffer: %d\n", err); goto cleanup; }
    Z_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating Z_buffer: %d\n", err); goto cleanup; }
    scale_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating scale_buffer: %d\n", err); goto cleanup; }
    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, 2 * dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating bounds_buffer: %d\n", err); goto cleanup; }

    // Initialize scale and bounds
    scale = (float *)malloc(dim * sizeof(float));
    if (!scale) { fprintf(stderr, "Error allocating scale\n"); goto cleanup; }
    bounds = (float *)malloc(2 * dim * sizeof(float));
    if (!bounds) { fprintf(stderr, "Error allocating bounds\n"); goto cleanup; }
    for (int j = 0; j < dim; j++) {
        scale[j] = (float)(opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        bounds[2 * j] = (float)opt->bounds[2 * j];
        bounds[2 * j + 1] = (float)opt->bounds[2 * j + 1];
    }
    err = clEnqueueWriteBuffer(opt->queue, scale_buffer, CL_TRUE, 0, dim * sizeof(float), scale, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error writing scale_buffer: %d\n", err); goto cleanup; }
    err = clEnqueueWriteBuffer(opt->queue, bounds_buffer, CL_TRUE, 0, 2 * dim * sizeof(float), bounds, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error writing bounds_buffer: %d\n", err); goto cleanup; }

    // Set kernel arguments for update_kernel
    err = clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &opt->population_buffer); // best_position
    err |= clSetKernelArg(update_kernel, 2, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(update_kernel, 3, sizeof(cl_mem), &Q_buffer);
    err |= clSetKernelArg(update_kernel, 4, sizeof(cl_mem), &v_buffer);
    err |= clSetKernelArg(update_kernel, 5, sizeof(cl_mem), &S_buffer);
    err |= clSetKernelArg(update_kernel, 6, sizeof(cl_mem), &M_buffer);
    err |= clSetKernelArg(update_kernel, 7, sizeof(cl_mem), &F_buffer);
    err |= clSetKernelArg(update_kernel, 8, sizeof(cl_mem), &T_buffer);
    err |= clSetKernelArg(update_kernel, 9, sizeof(cl_mem), &C_buffer);
    err |= clSetKernelArg(update_kernel, 10, sizeof(cl_mem), &V_buffer);
    err |= clSetKernelArg(update_kernel, 11, sizeof(cl_mem), &Z_buffer);
    err |= clSetKernelArg(update_kernel, 12, sizeof(cl_mem), &scale_buffer);
    err |= clSetKernelArg(update_kernel, 13, sizeof(int), &dim);
    err |= clSetKernelArg(update_kernel, 14, sizeof(int), &population_size);
    err |= clSetKernelArg(update_kernel, 15, sizeof(float), &alpha);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error setting update_kernel args: %d\n", err); goto cleanup; }

    // Set kernel arguments for fitness_kernel
    err = clSetKernelArg(fitness_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(fitness_kernel, 1, sizeof(cl_mem), &opt->fitness_buffer);
    err |= clSetKernelArg(fitness_kernel, 2, sizeof(cl_mem), &opt->population_buffer); // best_position
    err |= clSetKernelArg(fitness_kernel, 3, sizeof(cl_mem), &scale_buffer);
    err |= clSetKernelArg(fitness_kernel, 4, sizeof(int), &dim);
    err |= clSetKernelArg(fitness_kernel, 5, sizeof(int), &population_size);
    err |= clSetKernelArg(fitness_kernel, 6, sizeof(float), &alpha);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error setting fitness_kernel args: %d\n", err); goto cleanup; }

    // Allocate host buffers
    population_float = (float *)malloc(dim * population_size * sizeof(float));
    if (!population_float) { fprintf(stderr, "Error allocating population_float\n"); goto cleanup; }
    fitness_float = (float *)malloc(population_size * sizeof(float));
    if (!fitness_float) { fprintf(stderr, "Error allocating fitness_float\n"); goto cleanup; }

    // Verify population memory
    for (int i = 0; i < population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: opt->population[%d].position is NULL\n", i);
            goto cleanup;
        }
    }
    if (!opt->best_solution.position) {
        fprintf(stderr, "Error: opt->best_solution.position is NULL\n");
        goto cleanup;
    }

    // Main optimization loop
    size_t global_work_size = ((population_size + local_work_size - 1) / local_work_size) * local_work_size;
    *history_size = 0;
    int iter = 0;
    while (opt->best_solution.fitness > CA_TOL && iter < max_iter) {
        // Update alpha
        alpha *= 0.97f;
        unsigned int seed = (unsigned int)time(NULL) + iter;

        // Run cricket update kernel
        err = clSetKernelArg(update_kernel, 16, sizeof(unsigned int), &seed);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error setting update_kernel seed: %d\n", err); goto cleanup; }
        err = clEnqueueNDRangeKernel(opt->queue, update_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error enqueuing update_kernel: %d\n", err); goto cleanup; }

        // Read population for fitness evaluation
        err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, dim * population_size * sizeof(float), population_float, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error reading population_buffer: %d\n", err); goto cleanup; }
        for (int i = 0; i < population_size; i++) {
            double position[32]; // Assumes dim <= 32
            for (int j = 0; j < dim; j++) {
                position[j] = (double)population_float[i * dim + j];
            }
            double new_fitness = objective_function(position);
            fitness_float[i] = (float)new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < dim; j++) {
                    opt->best_solution.position[j] = position[j];
                }
                // Store history
                if (*history_size < max_history) {
                    history[*history_size] = (double *)malloc((dim + 1) * sizeof(double));
                    if (!history[*history_size]) {
                        fprintf(stderr, "Error allocating history entry\n");
                        goto cleanup;
                    }
                    for (int j = 0; j < dim; j++) {
                        history[*history_size][j] = opt->best_solution.position[j];
                    }
                    history[*history_size][dim] = opt->best_solution.fitness;
                    (*history_size)++;
                }
            }
        }
        err = clEnqueueWriteBuffer(opt->queue, opt->fitness_buffer, CL_TRUE, 0, population_size * sizeof(float), fitness_float, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error writing fitness_buffer: %d\n", err); goto cleanup; }

        // Run fitness update kernel
        err = clSetKernelArg(fitness_kernel, 7, sizeof(unsigned int), &seed);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error setting fitness_kernel seed: %d\n", err); goto cleanup; }
        err = clEnqueueNDRangeKernel(opt->queue, fitness_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error enqueuing fitness_kernel: %d\n", err); goto cleanup; }

        // Enforce bounds
        err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, dim * population_size * sizeof(float), population_float, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error reading population_buffer for bounds: %d\n", err); goto cleanup; }
        for (int i = 0; i < population_size; i++) {
            for (int j = 0; j < dim; j++) {
                opt->population[i].position[j] = (double)population_float[i * dim + j];
                if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                    opt->population[i].position[j] = opt->bounds[2 * j];
                } else if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                    opt->population[i].position[j] = opt->bounds[2 * j + 1];
                }
            }
        }
        for (int i = 0; i < population_size; i++) {
            for (int j = 0; j < dim; j++) {
                population_float[i * dim + j] = (float)opt->population[i].position[j];
            }
        }
        err = clEnqueueWriteBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, dim * population_size * sizeof(float), population_float, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "Error writing population_buffer: %d\n", err); goto cleanup; }

        iter++;
        printf("Iteration %d: Best Fitness = %f\n", iter, opt->best_solution.fitness);
    }

    // Final population read
    err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, dim * population_size * sizeof(float), population_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error reading final population_buffer: %d\n", err); goto cleanup; }

    // Debug print before cleanup
    fprintf(stderr, "Finished optimization loop, starting cleanup\n");

cleanup:
    if (population_float) free(population_float);
    if (fitness_float) free(fitness_float);
    if (scale) free(scale);
    if (bounds) free(bounds);
    population_float = NULL;
    fitness_float = NULL;
    scale = NULL;
    bounds = NULL;

    safe_release_cl_resources(program, update_kernel, fitness_kernel, Q_buffer, v_buffer, S_buffer, M_buffer,
                             F_buffer, T_buffer, C_buffer, V_buffer, Z_buffer, scale_buffer, bounds_buffer);
    program = NULL;
    update_kernel = NULL;
    fitness_kernel = NULL;
    Q_buffer = NULL;
    v_buffer = NULL;
    S_buffer = NULL;
    M_buffer = NULL;
    F_buffer = NULL;
    T_buffer = NULL;
    C_buffer = NULL;
    V_buffer = NULL;
    Z_buffer = NULL;
    scale_buffer = NULL;
    bounds_buffer = NULL;

    clFinish(opt->queue);
    fprintf(stderr, "Cleanup completed\n");
}
