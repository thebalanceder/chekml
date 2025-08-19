#include "GlowSO.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Convert objective function value to minimization form
static inline float convert_to_min(float fcn) {
    return (fcn >= 0.0f) ? 1.0f / (1.0f + fcn) : 1.0f + fabs(fcn);
}

// Preprocess kernel source (minimal for GlowSO)
char *preprocess_kernel_source(const char *source);

void GlowSO_init_cl(GlowSOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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

    // Reuse OpenCL context and device
    cl_ctx->context = opt->context;
    cl_ctx->device = opt->device;

    // Preprocess kernel source
    char *processed_source = preprocess_kernel_source(kernel_source);

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
            fprintf(stderr, "Error building program: %d\nBuild Log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(cl_ctx->program);
        free(processed_source);
        exit(1);
    }

    free(processed_source);

    // Create kernels
    cl_ctx->init_pop_kernel = clCreateKernel(cl_ctx->program, "initialize_population", &err);
    check_cl_error(err, "clCreateKernel init_pop");
    cl_ctx->luciferin_kernel = clCreateKernel(cl_ctx->program, "update_luciferin", &err);
    check_cl_error(err, "clCreateKernel luciferin");
    cl_ctx->movement_kernel = clCreateKernel(cl_ctx->program, "movement_phase", &err);
    check_cl_error(err, "clCreateKernel movement");
    cl_ctx->range_kernel = clCreateKernel(cl_ctx->program, "update_decision_range", &err);
    check_cl_error(err, "clCreateKernel range");
}

void GlowSO_cleanup_cl(GlowSOCLContext *cl_ctx) {
    if (cl_ctx->range_kernel) clReleaseKernel(cl_ctx->range_kernel);
    if (cl_ctx->movement_kernel) clReleaseKernel(cl_ctx->movement_kernel);
    if (cl_ctx->luciferin_kernel) clReleaseKernel(cl_ctx->luciferin_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void GlowSO_init_context(GlowSOContext *ctx, Optimizer *opt, GlowSOCLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;

    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->population = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer population");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->decision_range = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer decision_range");
    ctx->distances = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer distances");
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
}

void GlowSO_cleanup_context(GlowSOContext *ctx, GlowSOCLContext *cl_ctx) {
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->distances) clReleaseMemObject(ctx->distances);
    if (ctx->decision_range) clReleaseMemObject(ctx->decision_range);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

void GlowSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Glowworm Swarm Optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("GlowSO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open GlowSO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: GlowSO.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read GlowSO.cl completely\n");
        free(source_str);
        exit(1);
    }
    source_str[source_size] = '\0';

    // Initialize OpenCL context
    GlowSOCLContext cl_ctx = {0};
    GlowSO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    GlowSOContext ctx = {0};
    GlowSO_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * pop_size : MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_pop_time;
        double luciferin_time;
        double movement_time;
        double range_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;
    cl_mem best_fitness_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_fitness");
    cl_mem objective_values = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer objective_values");

    // Initialize population
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_pop_kernel, 0, sizeof(cl_mem), &ctx.population);
    clSetKernelArg(cl_ctx.init_pop_kernel, 1, sizeof(cl_mem), &ctx.fitness);
    clSetKernelArg(cl_ctx.init_pop_kernel, 2, sizeof(cl_mem), &ctx.decision_range);
    clSetKernelArg(cl_ctx.init_pop_kernel, 3, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 5, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_pop_kernel, 6, sizeof(cl_int), &pop_size);

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
                prof.init_pop_time += (end - start) / 1e6; // ns to ms
            }
        }
        clReleaseEvent(init_event);
    }

    // Main optimization loop
    while (func_count < max_evals) {
        iteration++;
        // Evaluate population on CPU
        cl_event read_event, write_event;
        float *positions = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population");
        float *obj_vals = (float *)malloc(pop_size * sizeof(float));
        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            obj_vals[i] = (float)objective_function(pos);
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, objective_values, CL_TRUE, 0, pop_size * sizeof(cl_float), obj_vals, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer objective_values");
        func_count += pop_size;

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

        // Update best solution
        float best_obj_val = obj_vals[0];
        int best_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (obj_vals[i] < best_obj_val) {
                best_obj_val = obj_vals[i];
                best_idx = i;
            }
        }
        float new_best_fitness = convert_to_min(best_obj_val);
        if (new_best_fitness < ctx.best_fitness) {
            ctx.best_fitness = new_best_fitness;
            float *pos = (float *)malloc(dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), pos, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer best_pos");
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), pos, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer best_pos");
            free(pos);

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

            err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &new_best_fitness, 0, NULL, &write_event);
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
        free(obj_vals);

        // Update luciferin
        cl_event luciferin_event;
        clSetKernelArg(cl_ctx.luciferin_kernel, 0, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.luciferin_kernel, 1, sizeof(cl_mem), &objective_values);
        clSetKernelArg(cl_ctx.luciferin_kernel, 2, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.luciferin_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &luciferin_event);
        check_cl_error(err, "clEnqueueNDRangeKernel luciferin");
        clFinish(cl_ctx.queue);

        if (luciferin_event) {
            err = clGetEventProfilingInfo(luciferin_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(luciferin_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.luciferin_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(luciferin_event);
        }

        // Movement phase
        cl_event movement_event;
        clSetKernelArg(cl_ctx.movement_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.movement_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.movement_kernel, 2, sizeof(cl_mem), &ctx.decision_range);
        clSetKernelArg(cl_ctx.movement_kernel, 3, sizeof(cl_mem), &ctx.distances);
        clSetKernelArg(cl_ctx.movement_kernel, 4, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.movement_kernel, 5, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.movement_kernel, 6, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.movement_kernel, 7, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.movement_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &movement_event);
        check_cl_error(err, "clEnqueueNDRangeKernel movement");
        clFinish(cl_ctx.queue);

        if (movement_event) {
            err = clGetEventProfilingInfo(movement_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(movement_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.movement_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(movement_event);
        }

        // Decision range update
        cl_event range_event;
        clSetKernelArg(cl_ctx.range_kernel, 0, sizeof(cl_mem), &ctx.decision_range);
        clSetKernelArg(cl_ctx.range_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.range_kernel, 2, sizeof(cl_mem), &ctx.distances);
        clSetKernelArg(cl_ctx.range_kernel, 3, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.range_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &range_event);
        check_cl_error(err, "clEnqueueNDRangeKernel range");
        clFinish(cl_ctx.queue);

        if (range_event) {
            err = clGetEventProfilingInfo(range_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(range_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.range_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(range_event);
        }

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", 
               iteration, func_count, ctx.best_fitness);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_pop_time);
    printf("Update Luciferin: %.3f\n", prof.luciferin_time);
    printf("Movement Phase: %.3f\n", prof.movement_time);
    printf("Decision Range Update: %.3f\n", prof.range_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Update Luciferin: %.3f\n", prof.luciferin_time / prof.count);
        printf("Movement Phase: %.3f\n", prof.movement_time / prof.count);
        printf("Decision Range Update: %.3f\n", prof.range_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Copy best solution to optimizer
    cl_event read_event;
    float *best_pos = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer final best_pos");
    opt->best_solution.fitness = ctx.best_fitness;
    for (int i = 0; i < dim; i++) opt->best_solution.position[i] = best_pos[i];
    free(best_pos);

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

    // Cleanup
    clReleaseMemObject(objective_values);
    clReleaseMemObject(best_fitness_buf);
    GlowSO_cleanup_context(&ctx, &cl_ctx);
    GlowSO_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
