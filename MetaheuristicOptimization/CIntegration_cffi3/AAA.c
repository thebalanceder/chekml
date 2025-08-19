#include "AAA.h"
#include "generaloptimizer.h"
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// OpenCL kernel source for AAA
static const char* aaa_kernel_source = 
"// Xorshift random number generator for GPU\n"
"unsigned int xorshift32(unsigned int seed) {\n"
"    unsigned int x = seed;\n"
"    x ^= x << 13;\n"
"    x ^= x >> 17;\n"
"    x ^= x << 5;\n"
"    return x;\n"
"}\n"
"\n"
"// Initialize population on GPU\n"
"__kernel void initialize_population(\n"
"    __global float* population,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const unsigned int seed)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        unsigned int local_seed = seed + id;\n"
"        for (int j = 0; j < dim; j++) {\n"
"            local_seed = xorshift32(local_seed);\n"
"            float lower = bounds[2 * j];\n"
"            float upper = bounds[2 * j + 1];\n"
"            float rand_val = (float)local_seed / 0xffffffffU;\n"
"            population[id * dim + j] = lower + rand_val * (upper - lower);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"// Movement phase with boundary enforcement\n"
"__kernel void movement_phase(\n"
"    __global float* population,\n"
"    __global const float* best_pos,\n"
"    __global const float* bounds,\n"
"    const int dim,\n"
"    const int population_size,\n"
"    const float step_size)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        for (int j = 0; j < dim; j++) {\n"
"            float direction = best_pos[j] - population[id * dim + j];\n"
"            population[id * dim + j] += step_size * direction;\n"
"            // Enforce boundaries\n"
"            float lower = bounds[2 * j];\n"
"            float upper = bounds[2 * j + 1];\n"
"            if (population[id * dim + j] < lower) population[id * dim + j] = lower;\n"
"            else if (population[id * dim + j] > upper) population[id * dim + j] = upper;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"// Evaluate Schwefel function\n"
"__kernel void evaluate_schwefel(\n"
"    __global const float* population,\n"
"    __global float* fitness,\n"
"    const int population_size,\n"
"    const int dim)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < population_size) {\n"
"        float result = 0.0f;\n"
"        for (int i = 0; i < dim; i++) {\n"
"            float x = population[id * dim + i];\n"
"            result += x * sin(sqrt(fabs(x)));\n"
"        }\n"
"        fitness[id] = 418.9829f * dim - result;\n"
"    }\n"
"}\n";

// Main Optimization Function
void AAA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel init_kernel = NULL, movement_kernel = NULL, eval_kernel = NULL;
    cl_mem bounds_buffer = NULL, best_pos_buffer = NULL;
    const float step_size = 0.1f;

    // Validate input pointers
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(EXIT_FAILURE);
    }

    // Create float array for bounds
    float* bounds_float = (float*)malloc(2 * opt->dim * sizeof(float));
    if (!bounds_float) {
        fprintf(stderr, "Error: Memory allocation failed for bounds_float\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < 2 * opt->dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }

    // Create OpenCL program for AAA-specific kernels
    program = clCreateProgramWithSource(opt->context, 1, &aaa_kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating AAA program: %d\n", err);
        free(bounds_float);
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building AAA program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        free(bounds_float);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    init_kernel = clCreateKernel(program, "initialize_population", &err);
    if (err != CL_SUCCESS || !init_kernel) {
        fprintf(stderr, "Error creating init kernel: %d\n", err);
        clReleaseProgram(program);
        free(bounds_float);
        exit(EXIT_FAILURE);
    }
    movement_kernel = clCreateKernel(program, "movement_phase", &err);
    if (err != CL_SUCCESS || !movement_kernel) {
        fprintf(stderr, "Error creating movement kernel: %d\n", err);
        clReleaseKernel(init_kernel);
        clReleaseProgram(program);
        free(bounds_float);
        exit(EXIT_FAILURE);
    }
    eval_kernel = clCreateKernel(program, "evaluate_schwefel", &err);
    if (err != CL_SUCCESS || !eval_kernel) {
        fprintf(stderr, "Error creating eval kernel: %d\n", err);
        clReleaseKernel(init_kernel);
        clReleaseKernel(movement_kernel);
        clReleaseProgram(program);
        free(bounds_float);
        exit(EXIT_FAILURE);
    }

    // Create buffer for bounds
    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, 
                                   2 * opt->dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer) {
        fprintf(stderr, "Error creating bounds buffer: %d\n", err);
        clReleaseKernel(init_kernel);
        clReleaseKernel(movement_kernel);
        clReleaseKernel(eval_kernel);
        clReleaseProgram(program);
        free(bounds_float);
        exit(EXIT_FAILURE);
    }

    // Write bounds to GPU
    err = clEnqueueWriteBuffer(opt->queue, bounds_buffer, CL_TRUE, 0, 
                               2 * opt->dim * sizeof(float), bounds_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bounds to buffer: %d\n", err);
        clReleaseMemObject(bounds_buffer);
        clReleaseKernel(init_kernel);
        clReleaseKernel(movement_kernel);
        clReleaseKernel(eval_kernel);
        clReleaseProgram(program);
        free(bounds_float);
        exit(EXIT_FAILURE);
    }

    // Initialize population on GPU
    size_t global_work_size = opt->population_size;
    unsigned int seed = (unsigned int)time(NULL);
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(init_kernel, 2, sizeof(int), &opt->dim);
    err |= clSetKernelArg(init_kernel, 3, sizeof(int), &opt->population_size);
    err |= clSetKernelArg(init_kernel, 4, sizeof(unsigned int), &seed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting init kernel args: %d\n", err);
        clReleaseMemObject(bounds_buffer);
        clReleaseKernel(init_kernel);
        clReleaseKernel(movement_kernel);
        clReleaseKernel(eval_kernel);
        clReleaseProgram(program);
        free(bounds_float);
        exit(EXIT_FAILURE);
    }

    err = clEnqueueNDRangeKernel(opt->queue, init_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing init kernel: %d\n", err);
        clReleaseMemObject(bounds_buffer);
        clReleaseKernel(init_kernel);
        clReleaseKernel(movement_kernel);
        clReleaseKernel(eval_kernel);
        clReleaseProgram(program);
        free(bounds_float);
        exit(EXIT_FAILURE);
    }

    // Create buffer for best position
    best_pos_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, 
                                     opt->dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !best_pos_buffer) {
        fprintf(stderr, "Error creating best_pos buffer: %d\n", err);
        clReleaseMemObject(bounds_buffer);
        clReleaseKernel(init_kernel);
        clReleaseKernel(movement_kernel);
        clReleaseKernel(eval_kernel);
        clReleaseProgram(program);
        free(bounds_float);
        exit(EXIT_FAILURE);
    }

    // Initialize best solution (convert double to float)
    float* best_pos_float = (float*)malloc(opt->dim * sizeof(float));
    if (!best_pos_float) {
        fprintf(stderr, "Error: Memory allocation failed for best_pos_float\n");
        clReleaseMemObject(bounds_buffer);
        clReleaseMemObject(best_pos_buffer);
        clReleaseKernel(init_kernel);
        clReleaseKernel(movement_kernel);
        clReleaseKernel(eval_kernel);
        clReleaseProgram(program);
        free(bounds_float);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < opt->dim; i++) {
        best_pos_float[i] = (float)opt->best_solution.position[i];
    }
    opt->best_solution.fitness = INFINITY;
    err = clEnqueueWriteBuffer(opt->queue, best_pos_buffer, CL_TRUE, 0, 
                               opt->dim * sizeof(float), best_pos_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing best_pos to buffer: %d\n", err);
        clReleaseMemObject(best_pos_buffer);
        clReleaseMemObject(bounds_buffer);
        clReleaseKernel(init_kernel);
        clReleaseKernel(movement_kernel);
        clReleaseKernel(eval_kernel);
        clReleaseProgram(program);
        free(bounds_float);
        free(best_pos_float);
        exit(EXIT_FAILURE);
    }

    // Main optimization loop
    float* temp_fitness = (float*)malloc(opt->population_size * sizeof(float));
    float* temp_best_pos = (float*)malloc(opt->dim * sizeof(float));
    if (!temp_fitness || !temp_best_pos) {
        fprintf(stderr, "Error: Memory allocation failed for temp_fitness or temp_best_pos\n");
        clReleaseMemObject(best_pos_buffer);
        clReleaseMemObject(bounds_buffer);
        clReleaseKernel(init_kernel);
        clReleaseKernel(movement_kernel);
        clReleaseKernel(eval_kernel);
        clReleaseProgram(program);
        free(bounds_float);
        free(best_pos_float);
        free(temp_fitness);
        free(temp_best_pos);
        exit(EXIT_FAILURE);
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate population on GPU
        err = clSetKernelArg(eval_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(eval_kernel, 1, sizeof(cl_mem), &opt->fitness_buffer);
        err |= clSetKernelArg(eval_kernel, 2, sizeof(int), &opt->population_size);
        err |= clSetKernelArg(eval_kernel, 3, sizeof(int), &opt->dim);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting eval kernel args: %d\n", err);
            clReleaseMemObject(best_pos_buffer);
            clReleaseMemObject(bounds_buffer);
            clReleaseKernel(init_kernel);
            clReleaseKernel(movement_kernel);
            clReleaseKernel(eval_kernel);
            clReleaseProgram(program);
            free(bounds_float);
            free(best_pos_float);
            free(temp_fitness);
            free(temp_best_pos);
            exit(EXIT_FAILURE);
        }

        err = clEnqueueNDRangeKernel(opt->queue, eval_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing eval kernel: %d\n", err);
            clReleaseMemObject(best_pos_buffer);
            clReleaseMemObject(bounds_buffer);
            clReleaseKernel(init_kernel);
            clReleaseKernel(movement_kernel);
            clReleaseKernel(eval_kernel);
            clReleaseProgram(program);
            free(bounds_float);
            free(best_pos_float);
            free(temp_fitness);
            free(temp_best_pos);
            exit(EXIT_FAILURE);
        }

        // Read fitness values
        err = clEnqueueReadBuffer(opt->queue, opt->fitness_buffer, CL_TRUE, 0, 
                                  opt->population_size * sizeof(float), temp_fitness, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading fitness buffer: %d\n", err);
            clReleaseMemObject(best_pos_buffer);
            clReleaseMemObject(bounds_buffer);
            clReleaseKernel(init_kernel);
            clReleaseKernel(movement_kernel);
            clReleaseKernel(eval_kernel);
            clReleaseProgram(program);
            free(bounds_float);
            free(best_pos_float);
            free(temp_fitness);
            free(temp_best_pos);
            exit(EXIT_FAILURE);
        }

        // Find best solution
        double best_fitness = opt->best_solution.fitness;
        int best_idx = -1;
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = (double)temp_fitness[i];
            if (temp_fitness[i] < best_fitness) {
                best_fitness = temp_fitness[i];
                best_idx = i;
            }
        }

        if (best_idx >= 0) {
            opt->best_solution.fitness = best_fitness;
            // Read best position from population buffer
            err = clEnqueueReadBuffer(opt->queue, opt->population_buffer, CL_TRUE, 
                                      best_idx * opt->dim * sizeof(float), 
                                      opt->dim * sizeof(float), temp_best_pos, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading best position: %d\n", err);
                clReleaseMemObject(best_pos_buffer);
                clReleaseMemObject(bounds_buffer);
                clReleaseKernel(init_kernel);
                clReleaseKernel(movement_kernel);
                clReleaseKernel(eval_kernel);
                clReleaseProgram(program);
                free(bounds_float);
                free(best_pos_float);
                free(temp_fitness);
                free(temp_best_pos);
                exit(EXIT_FAILURE);
            }
            // Convert float to double for best_solution.position
            for (int i = 0; i < opt->dim; i++) {
                opt->best_solution.position[i] = (double)temp_best_pos[i];
            }
            // Update best_pos_buffer
            for (int i = 0; i < opt->dim; i++) {
                best_pos_float[i] = temp_best_pos[i];
            }
            err = clEnqueueWriteBuffer(opt->queue, best_pos_buffer, CL_TRUE, 0, 
                                       opt->dim * sizeof(float), best_pos_float, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error writing best_pos to buffer: %d\n", err);
                clReleaseMemObject(best_pos_buffer);
                clReleaseMemObject(bounds_buffer);
                clReleaseKernel(init_kernel);
                clReleaseKernel(movement_kernel);
                clReleaseKernel(eval_kernel);
                clReleaseProgram(program);
                free(bounds_float);
                free(best_pos_float);
                free(temp_fitness);
                free(temp_best_pos);
                exit(EXIT_FAILURE);
            }
        }

        // Movement phase on GPU
        err = clSetKernelArg(movement_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
        err |= clSetKernelArg(movement_kernel, 1, sizeof(cl_mem), &best_pos_buffer);
        err |= clSetKernelArg(movement_kernel, 2, sizeof(cl_mem), &bounds_buffer);
        err |= clSetKernelArg(movement_kernel, 3, sizeof(int), &opt->dim);
        err |= clSetKernelArg(movement_kernel, 4, sizeof(int), &opt->population_size);
        err |= clSetKernelArg(movement_kernel, 5, sizeof(float), &step_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error setting movement kernel args: %d\n", err);
            clReleaseMemObject(best_pos_buffer);
            clReleaseMemObject(bounds_buffer);
            clReleaseKernel(init_kernel);
            clReleaseKernel(movement_kernel);
            clReleaseKernel(eval_kernel);
            clReleaseProgram(program);
            free(bounds_float);
            free(best_pos_float);
            free(temp_fitness);
            free(temp_best_pos);
            exit(EXIT_FAILURE);
        }

        err = clEnqueueNDRangeKernel(opt->queue, movement_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error enqueuing movement kernel: %d\n", err);
            clReleaseMemObject(best_pos_buffer);
            clReleaseMemObject(bounds_buffer);
            clReleaseKernel(init_kernel);
            clReleaseKernel(movement_kernel);
            clReleaseKernel(eval_kernel);
            clReleaseProgram(program);
            free(bounds_float);
            free(best_pos_float);
            free(temp_fitness);
            free(temp_best_pos);
            exit(EXIT_FAILURE);
        }

        // Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Cleanup
    free(temp_fitness);
    free(temp_best_pos);
    free(best_pos_float);
    free(bounds_float);
    clReleaseMemObject(bounds_buffer);
    clReleaseMemObject(best_pos_buffer);
    clReleaseKernel(init_kernel);
    clReleaseKernel(movement_kernel);
    clReleaseKernel(eval_kernel);
    clReleaseProgram(program);
}
