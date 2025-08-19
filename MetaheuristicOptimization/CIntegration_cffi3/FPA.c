#include "FPA.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (placeholder)
char *preprocess_kernel_source(const char *source);

void fpa_init_cl(FPA_CLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->init_pop_kernel = clCreateKernel(cl_ctx->program, "initialize_population", &err);
    check_cl_error(err, "clCreateKernel init_pop");
    cl_ctx->global_poll_kernel = clCreateKernel(cl_ctx->program, "global_pollination_phase", &err);
    check_cl_error(err, "clCreateKernel global_poll");
    cl_ctx->local_poll_kernel = clCreateKernel(cl_ctx->program, "local_pollination_phase", &err);
    check_cl_error(err, "clCreateKernel local_poll");
}

void fpa_cleanup_cl(FPA_CLContext *cl_ctx) {
    if (cl_ctx->local_poll_kernel) clReleaseKernel(cl_ctx->local_poll_kernel);
    if (cl_ctx->global_poll_kernel) clReleaseKernel(cl_ctx->global_poll_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void fpa_init_context(FPA_Context *ctx, Optimizer *opt, FPA_CLContext *cl_ctx) {
    cl_int err;
    cl_int pop_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->population = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer population");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->best_position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_position");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");

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

    ctx->best_fitness = INFINITY;
}

void fpa_cleanup_context(FPA_Context *ctx, FPA_CLContext *cl_ctx) {
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

void FPA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Flower Pollination Optimization...\n\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device ||
        !opt->population_buffer || !opt->fitness_buffer ||
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("FPA.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open FPA.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: FPA.cl is empty\n");
        fclose(fp);
        exit(1);
    }
    rewind(fp);
    char *source = (char *)malloc(source_size + 1);
    if (!source) {
        fprintf(stderr, "Error: Memory allocation failed for kernel source\n");
        fclose(fp);
        exit(1);
    }
    size_t read_size = fread(source, 1, source_size, fp);
    fclose(fp);
    if (read_size != source_size) {
        fprintf(stderr, "Error: Failed to read FPA.cl completely: read %zu, expected %zu\n", read_size, source_size);
        free(source);
        exit(1);
    }
    source[source_size] = '\0';

    // Initialize OpenCL context
    FPA_CLContext cl_ctx = {0};
    fpa_init_cl(&cl_ctx, source, opt);
    free(source);

    // Initialize algorithm context
    FPA_Context ctx = {0};
    fpa_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * pop_size : FPA_MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_pop_time;
        double global_poll_time;
        double local_poll_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;
    cl_mem best_fitness_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_fitness");

    // Initialize population
    cl_event init_event;
    err = clSetKernelArg(cl_ctx.init_pop_kernel, 0, sizeof(cl_mem), &ctx.population);
    check_cl_error(err, "clSetKernelArg init_pop 0");
    err = clSetKernelArg(cl_ctx.init_pop_kernel, 1, sizeof(cl_mem), &ctx.fitness);
    check_cl_error(err, "clSetKernelArg init_pop 1");
    err = clSetKernelArg(cl_ctx.init_pop_kernel, 2, sizeof(cl_mem), &ctx.bounds);
    check_cl_error(err, "clSetKernelArg init_pop 2");
    err = clSetKernelArg(cl_ctx.init_pop_kernel, 3, sizeof(cl_mem), &ctx.best_position);
    check_cl_error(err, "clSetKernelArg init_pop 3");
    err = clSetKernelArg(cl_ctx.init_pop_kernel, 4, sizeof(cl_mem), &best_fitness_buf);
    check_cl_error(err, "clSetKernelArg init_pop 4");
    err = clSetKernelArg(cl_ctx.init_pop_kernel, 5, sizeof(cl_mem), &ctx.random_seeds);
    check_cl_error(err, "clSetKernelArg init_pop 5");
    err = clSetKernelArg(cl_ctx.init_pop_kernel, 6, sizeof(cl_int), &dim);
    check_cl_error(err, "clSetKernelArg init_pop 6");
    err = clSetKernelArg(cl_ctx.init_pop_kernel, 7, sizeof(cl_int), &pop_size);
    check_cl_error(err, "clSetKernelArg init_pop 7");

    size_t global_work_size = pop_size;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.init_pop_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
    check_cl_error(err, "clEnqueueNDRangeKernel init_pop");
    clFinish(cl_ctx.queue);

    cl_ulong start, end;
    if (init_event) {
        err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.init_pop_time += (end - start) / 1e6; // Convert ns to ms
            }
        }
        clReleaseEvent(init_event);
    }

    // Evaluate initial population on CPU
    cl_event read_event, write_event;
    float *positions = (float *)malloc(pop_size * dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer population");
    float *fitness = (float *)malloc(pop_size * sizeof(float));
    for (int i = 0; i < pop_size; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
        fitness[i] = (float)objective_function(pos);
        free(pos);
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer fitness");

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
    float best_fitness = fitness[0];
    int best_idx = 0;
    for (int i = 1; i < pop_size; i++) {
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            best_idx = i;
        }
    }
    ctx.best_fitness = best_fitness;

    float *best_pos = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer best_pos");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer best_pos");

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

    err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &best_fitness, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer best_fitness");
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

    // Main optimization loop
    while (func_count < max_evals) {
        iteration++;
        printf("Iteration %d\n", iteration);

        // Global pollination phase
        cl_event global_poll_event;
        err = clSetKernelArg(cl_ctx.global_poll_kernel, 0, sizeof(cl_mem), &ctx.population);
        check_cl_error(err, "clSetKernelArg global_poll 0");
        err = clSetKernelArg(cl_ctx.global_poll_kernel, 1, sizeof(cl_mem), &ctx.bounds);
        check_cl_error(err, "clSetKernelArg global_poll 1");
        err = clSetKernelArg(cl_ctx.global_poll_kernel, 2, sizeof(cl_mem), &ctx.best_position);
        check_cl_error(err, "clSetKernelArg global_poll 2");
        err = clSetKernelArg(cl_ctx.global_poll_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
        check_cl_error(err, "clSetKernelArg global_poll 3");
        err = clSetKernelArg(cl_ctx.global_poll_kernel, 4, sizeof(cl_int), &dim);
        check_cl_error(err, "clSetKernelArg global_poll 4");
        err = clSetKernelArg(cl_ctx.global_poll_kernel, 5, sizeof(cl_int), &pop_size);
        check_cl_error(err, "clSetKernelArg global_poll 5");

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.global_poll_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &global_poll_event);
        check_cl_error(err, "clEnqueueNDRangeKernel global_poll");
        clFinish(cl_ctx.queue);

        double global_poll_time = 0.0;
        if (global_poll_event) {
            err = clGetEventProfilingInfo(global_poll_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(global_poll_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    global_poll_time = (end - start) / 1e6;
                    prof.global_poll_time += global_poll_time;
                }
            }
            clReleaseEvent(global_poll_event);
        }

        // Local pollination phase
        cl_event local_poll_event;
        err = clSetKernelArg(cl_ctx.local_poll_kernel, 0, sizeof(cl_mem), &ctx.population);
        check_cl_error(err, "clSetKernelArg local_poll 0");
        err = clSetKernelArg(cl_ctx.local_poll_kernel, 1, sizeof(cl_mem), &ctx.bounds);
        check_cl_error(err, "clSetKernelArg local_poll 1");
        err = clSetKernelArg(cl_ctx.local_poll_kernel, 2, sizeof(cl_mem), &ctx.random_seeds);
        check_cl_error(err, "clSetKernelArg local_poll 2");
        err = clSetKernelArg(cl_ctx.local_poll_kernel, 3, sizeof(cl_int), &dim);
        check_cl_error(err, "clSetKernelArg local_poll 3");
        err = clSetKernelArg(cl_ctx.local_poll_kernel, 4, sizeof(cl_int), &pop_size);
        check_cl_error(err, "clSetKernelArg local_poll 4");

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.local_poll_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &local_poll_event);
        check_cl_error(err, "clEnqueueNDRangeKernel local_poll");
        clFinish(cl_ctx.queue);

        double local_poll_time = 0.0;
        if (local_poll_event) {
            err = clGetEventProfilingInfo(local_poll_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(local_poll_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    local_poll_time = (end - start) / 1e6;
                    prof.local_poll_time += local_poll_time;
                }
            }
            clReleaseEvent(local_poll_event);
        }

        // Evaluate population
        positions = (float *)realloc(positions, pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population");
        fitness = (float *)realloc(fitness, pop_size * sizeof(float));
        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            fitness[i] = (float)objective_function(pos);
            if (fitness[i] < ctx.best_fitness) {
                ctx.best_fitness = fitness[i];
                err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), &positions[i * dim], 0, NULL, &write_event);
                check_cl_error(err, "clEnqueueWriteBuffer best_pos");
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
                err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &fitness[i], 0, NULL, &write_event);
                check_cl_error(err, "clEnqueueWriteBuffer best_fitness");
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
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer fitness");
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

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.best_fitness);
        printf("Profiling (ms): Global Poll = %.3f, Local Poll = %.3f\n", global_poll_time, local_poll_time);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_pop_time);
    printf("Global Pollination Phase: %.3f\n", prof.global_poll_time);
    printf("Local Pollination Phase: %.3f\n", prof.local_poll_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("\nAverage per Iteration (ms):\n");
        printf("Global Pollination Phase: %.3f\n", prof.global_poll_time / prof.count);
        printf("Local Pollination Phase: %.3f\n", prof.local_poll_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Copy best solution to optimizer
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer final best_pos");
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

    opt->best_solution.fitness = ctx.best_fitness;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = best_pos[i];
    }
    free(best_pos);
    free(positions);
    free(fitness);

    // Cleanup
    clReleaseMemObject(best_fitness_buf);
    fpa_cleanup_context(&ctx, &cl_ctx);
    fpa_cleanup_cl(&cl_ctx);

    printf("\nOptimization completed.\n");
    printf("Total number of evaluations: %d\n", func_count);
    printf("Best solution: [");
    for (int j = 0; j < dim; j++) {
        printf("%f", opt->best_solution.position[j]);
        if (j < dim - 1) printf(", ");
    }
    printf("]\n");
    printf("Best value: %f\n", opt->best_solution.fitness);
}
