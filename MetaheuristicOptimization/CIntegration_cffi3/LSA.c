#include "LSA.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>

// Random double between min and max
static double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (placeholder)
char *preprocess_kernel_source(const char *source);

void LSA_init_cl(LSACLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
    cl_int err;

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }
    if (!kernel_source || strlen(kernel_source) == 0) {
        fprintf(stderr, "Error: Kernel source is empty or null\n");
        exit(1);
    }

    // Create command queue with profiling
    cl_command_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };
    cl_ctx->queue = clCreateCommandQueueWithProperties(opt->context, opt->device, props, &err);
    check_cl_error(err, "clCreateCommandQueueWithProperties");
    cl_ctx->owns_queue = CL_TRUE;

    // Log queue properties
    cl_command_queue_properties actual_props;
    err = clGetCommandQueueInfo(cl_ctx->queue, CL_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &actual_props, NULL);
    check_cl_error(err, "clGetCommandQueueInfo");
    printf("Queue profiling enabled: %s\n", (actual_props & CL_QUEUE_PROFILING_ENABLE) ? "Yes" : "No");

    // Preprocess kernel source
    char *processed_source = preprocess_kernel_source(kernel_source);

    // Reuse OpenCL context and device
    cl_ctx->context = opt->context;
    cl_ctx->device = opt->device;

    // Create and build program
    cl_ctx->program = clCreateProgramWithSource(cl_ctx->context, 1, (const char **)&processed_source, NULL, &err);
    check_cl_error(err, "clCreateProgramWithSource");

    const char *build_options = "-cl-std=CL1.2 -cl-mad-enable";
    err = clBuildProgram(cl_ctx->program, 1, &cl_ctx->device, build_options, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(cl_ctx->program, cl_ctx->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size + 1);
        if (log) {
            clGetProgramBuildInfo(cl_ctx->program, cl_ctx->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            log[log_size] = '\0';
            fprintf(stderr, "Error building program: %d\nDetailed Build Log:\n%s\n", err, log);
            free(log);
        } else {
            fprintf(stderr, "Error building program: %d\nFailed to allocate memory for build log\n", err);
        }
        clReleaseProgram(cl_ctx->program);
        free(processed_source);
        exit(1);
    }

    free(processed_source);

    // Create kernels
    cl_ctx->init_kernel = clCreateKernel(cl_ctx->program, "initialize_channels", &err);
    check_cl_error(err, "clCreateKernel init");
    cl_ctx->update_kernel = clCreateKernel(cl_ctx->program, "lsa_update_positions", &err);
    check_cl_error(err, "clCreateKernel update");
}

void LSA_cleanup_cl(LSACLContext *cl_ctx) {
    if (cl_ctx->update_kernel) clReleaseKernel(cl_ctx->update_kernel);
    if (cl_ctx->init_kernel) clReleaseKernel(cl_ctx->init_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void LSA_init_context(LSAContext *ctx, Optimizer *opt, LSACLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;
    ctx->channel_time = 0;

    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer position");
    ctx->directions = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer directions");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->best_position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_position");

    // Initialize random seeds
    cl_uint *seeds = (cl_uint *)malloc(pop_size * sizeof(cl_uint));
    srand(time(NULL));
    for (int i = 0; i < pop_size; i++) seeds[i] = rand();
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->random_seeds, CL_TRUE, 0, pop_size * sizeof(cl_uint), seeds, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer seeds");
    free(seeds);

    // Initialize bounds
    float *bounds_float = (float *)malloc(dim * 2 * sizeof(float));
    for (int i = 0; i < dim * 2; i++) bounds_float[i] = (float)opt->bounds[i];
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->bounds, CL_TRUE, 0, dim * 2 * sizeof(cl_float), bounds_float, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer bounds");
    free(bounds_float);

    // Initialize directions
    float *directions = (float *)malloc(dim * sizeof(float));
    for (int j = 0; j < dim; j++) directions[j] = (float)rand_double(-1.0, 1.0);
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->directions, CL_TRUE, 0, dim * sizeof(cl_float), directions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer directions");
    free(directions);
}

void LSA_cleanup_context(LSAContext *ctx, LSACLContext *cl_ctx) {
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->directions) clReleaseMemObject(ctx->directions);
    if (ctx->position) clReleaseMemObject(ctx->position);
}

// CPU-based direction update (simplified)
void update_directions_cpu(LSAContext *ctx, Optimizer *opt, ObjectiveFunction objective_function, int best_idx, LSACLContext *cl_ctx) {
    cl_int err;
    int dim = opt->dim;
    float *positions = (float *)malloc(opt->population_size * dim * sizeof(float));
    float *directions = (float *)malloc(dim * sizeof(float));
    float *test_channel = (float *)malloc(dim * sizeof(float));

    // Read positions and directions
    err = clEnqueueReadBuffer(cl_ctx->queue, ctx->position, CL_TRUE, 0, opt->population_size * dim * sizeof(cl_float), positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueReadBuffer position");
    err = clEnqueueReadBuffer(cl_ctx->queue, ctx->directions, CL_TRUE, 0, dim * sizeof(cl_float), directions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueReadBuffer directions");

    float best_fitness = ctx->best_fitness;
    for (int j = 0; j < dim; j++) {
        test_channel[j] = positions[best_idx * dim + j];
    }

    for (int j = 0; j < dim; j++) {
        float lb = (float)opt->bounds[2 * j];
        float ub = (float)opt->bounds[2 * j + 1];
        test_channel[j] += directions[j] * DIRECTION_STEP * (ub - lb);
        double test_channel_d[dim];
        for (int k = 0; k < dim; k++) test_channel_d[k] = (double)test_channel[k];
        float test_fitness = (float)objective_function(test_channel_d);
        if (test_fitness >= best_fitness) {
            directions[j] = -directions[j];
        }
        test_channel[j] = positions[best_idx * dim + j];
    }

    // Write updated directions
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->directions, CL_TRUE, 0, dim * sizeof(cl_float), directions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer directions");

    free(positions);
    free(directions);
    free(test_channel);
}

// CPU-based channel elimination
void update_channel_elimination_cpu(LSAContext *ctx, Optimizer *opt, LSACLContext *cl_ctx) {
    ctx->channel_time++;
    if (ctx->channel_time < MAX_CHANNEL_TIME) return;

    cl_int err;
    int pop_size = opt->population_size;
    int dim = opt->dim;
    float *fitness = (float *)malloc(pop_size * sizeof(float));
    float *positions = (float *)malloc(pop_size * dim * sizeof(float));

    // Read fitness and positions
    err = clEnqueueReadBuffer(cl_ctx->queue, ctx->fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueReadBuffer fitness");
    err = clEnqueueReadBuffer(cl_ctx->queue, ctx->position, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueReadBuffer position");

    // Find worst and best channels
    int worst_idx = 0, best_idx = 0;
    float worst_fitness = fitness[0], best_fitness = fitness[0];
    for (int i = 1; i < pop_size; i++) {
        if (fitness[i] > worst_fitness) {
            worst_fitness = fitness[i];
            worst_idx = i;
        }
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            best_idx = i;
        }
    }

    // Replace worst with best
    for (int j = 0; j < dim; j++) {
        positions[worst_idx * dim + j] = positions[best_idx * dim + j];
    }
    fitness[worst_idx] = fitness[best_idx];

    // Write back
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->position, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer position");
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer fitness");

    ctx->channel_time = 0;
    free(fitness);
    free(positions);
}

void LSA_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    printf("Starting GPU Lightning Search Algorithm...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device ||
        !opt->population_buffer || !opt->fitness_buffer ||
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("LSA.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open LSA.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: LSA.cl is empty\n");
        fclose(fp);
        exit(1);
    }
    rewind(fp);
    char *source_str = (char *)malloc(source_size + 1);
    if (!source_str) {
        fprintf(stderr, "Error: Memory allocation failed for kernel source\n");
        fclose(fp);
        exit(1);
    }
    size_t read_size = fread(source_str, 1, source_size, fp);
    fclose(fp);
    if (read_size != source_size) {
        fprintf(stderr, "Error: Failed to read LSA.cl completely (%zu of %zu bytes)\n", read_size, source_size);
        free(source_str);
        exit(1);
    }
    source_str[source_size] = '\0';

    int valid_content = 0;
    for (size_t i = 0; i < source_size; i++) {
        if (!isspace(source_str[i]) && isprint(source_str[i])) {
            valid_content = 1;
            break;
        }
    }
    if (!valid_content) {
        fprintf(stderr, "Error: LSA.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    LSACLContext cl_ctx = {0};
    LSA_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    LSAContext ctx = {0};
    LSA_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * pop_size : MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_time;
        double update_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;

    // Initialize channels
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_kernel, 0, sizeof(cl_mem), &ctx.position);
    clSetKernelArg(cl_ctx.init_kernel, 1, sizeof(cl_mem), &ctx.fitness);
    clSetKernelArg(cl_ctx.init_kernel, 2, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_kernel, 4, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_kernel, 5, sizeof(cl_int), &pop_size);

    size_t global_work_size = pop_size;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.init_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
    check_cl_error(err, "clEnqueueNDRangeKernel init");
    clFinish(cl_ctx.queue);

    cl_ulong start, end;
    if (init_event) {
        err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.init_time += (end - start) / 1e6; // Convert ns to ms
            }
        }
        clReleaseEvent(init_event);
    }

    // Evaluate initial population
    cl_event read_event, write_event;
    float *positions = (float *)malloc(pop_size * dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.position, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer position");
    float *fitness = (float *)malloc(pop_size * sizeof(float));
    int best_idx = 0;
    for (int i = 0; i < pop_size; i++) {
        double pos_d[dim];
        for (int j = 0; j < dim; j++) pos_d[j] = (double)positions[i * dim + j];
        fitness[i] = (float)objective_function(pos_d);
        if (i > 0 && fitness[i] < fitness[best_idx]) best_idx = i;
    }
    ctx.best_fitness = fitness[best_idx];
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer fitness");

    // Update best position
    float *best_pos = (float *)malloc(dim * sizeof(float));
    for (int j = 0; j < dim; j++) best_pos[j] = positions[best_idx * dim + j];
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer best_position");

    if (read_event) {
        err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.read_time += (end - start) / 1e6;
            }
        }
        clReleaseEvent(read_event);
    }

    if (write_event) {
        err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.write_time += (end - start) / 1e6;
            }
        }
        clReleaseEvent(write_event);
    }

    func_count += pop_size;

    // Main optimization loop
    while (func_count < max_evals) {
        iteration++;
        printf("Iteration %d\n", iteration);

        // Update channel elimination
        update_channel_elimination_cpu(&ctx, opt, &cl_ctx);

        // Update directions
        update_directions_cpu(&ctx, opt, objective_function, best_idx, &cl_ctx);

        // Update positions
        cl_event update_event;
        cl_float energy = LSA_ENERGY_FACTOR - 2.0f * exp(-5.0f * (opt->max_iter - iteration) / (float)opt->max_iter);
        clSetKernelArg(cl_ctx.update_kernel, 0, sizeof(cl_mem), &ctx.position);
        clSetKernelArg(cl_ctx.update_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.update_kernel, 2, sizeof(cl_mem), &ctx.directions);
        clSetKernelArg(cl_ctx.update_kernel, 3, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.update_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.update_kernel, 5, sizeof(cl_int), &best_idx);
        clSetKernelArg(cl_ctx.update_kernel, 6, sizeof(cl_float), &energy);
        clSetKernelArg(cl_ctx.update_kernel, 7, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.update_kernel, 8, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &update_event);
        check_cl_error(err, "clEnqueueNDRangeKernel update");
        clFinish(cl_ctx.queue);

        if (update_event) {
            err = clGetEventProfilingInfo(update_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(update_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.update_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(update_event);
        }

        // Evaluate fitness
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.position, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer position");
        float new_best_fitness = fitness[best_idx];
        int new_best_idx = best_idx;
        for (int i = 0; i < pop_size; i++) {
            double pos_d[dim];
            for (int j = 0; j < dim; j++) pos_d[j] = (double)positions[i * dim + j];
            fitness[i] = (float)objective_function(pos_d);
            if (fitness[i] < new_best_fitness) {
                new_best_fitness = fitness[i];
                new_best_idx = i;
            }
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer fitness");

        // Update best solution
        if (new_best_fitness < ctx.best_fitness) {
            ctx.best_fitness = new_best_fitness;
            best_idx = new_best_idx;
            for (int j = 0; j < dim; j++) best_pos[j] = positions[best_idx * dim + j];
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer best_position");
            if (write_event) {
                err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        prof.write_time += (end - start) / 1e6;
                    }
                }
                clReleaseEvent(write_event);
            }
        }

        if (read_event) {
            err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.read_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(read_event);
        }

        if (write_event) {
            err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.write_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(write_event);
        }

        func_count += pop_size;

        // Check convergence
        float best_fitness = fitness[0], worst_fitness = fitness[0];
        for (int i = 0; i < pop_size; i++) {
            if (fitness[i] < best_fitness) best_fitness = fitness[i];
            if (fitness[i] > worst_fitness) worst_fitness = fitness[i];
        }
        if (fabs(best_fitness - worst_fitness) < 1e-10) {
            printf("Converged at iteration %d\n", iteration);
            break;
        }

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.best_fitness);
    }

    // Copy best solution to optimizer
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer best_pos final");
    opt->best_solution.fitness = ctx.best_fitness;
    for (int i = 0; i < dim; i++) opt->best_solution.position[i] = (double)best_pos[i];
    free(best_pos);
    free(positions);
    free(fitness);

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialization: %.3f\n", prof.init_time);
    printf("Position Updates: %.3f\n", prof.update_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Position Updates: %.3f\n", prof.update_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Cleanup
    LSA_cleanup_context(&ctx, &cl_ctx);
    LSA_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
