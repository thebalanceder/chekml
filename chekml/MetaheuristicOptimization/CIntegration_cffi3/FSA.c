#include "FSA.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

#if FSA_DEBUG
#include <stdio.h>
#endif

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source to fix address space qualifiers
char *preprocess_kernel_source(const char *source);

void FSA_init_cl(FSACLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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

    // Create and build program
    char *processed_source = preprocess_kernel_source(kernel_source);
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
    cl_ctx->update_pop_kernel = clCreateKernel(cl_ctx->program, "update_population", &err);
    check_cl_error(err, "clCreateKernel update_population");
    cl_ctx->initial_strategy_kernel = clCreateKernel(cl_ctx->program, "initial_strategy_update", &err);
    check_cl_error(err, "clCreateKernel initial_strategy_update");
}

void FSA_cleanup_cl(FSACLContext *cl_ctx) {
    if (cl_ctx->initial_strategy_kernel) clReleaseKernel(cl_ctx->initial_strategy_kernel);
    if (cl_ctx->update_pop_kernel) clReleaseKernel(cl_ctx->update_pop_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void FSA_init_context(FSAContext *ctx, Optimizer *opt, FSACLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;

    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->population = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer population");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->local_best_positions = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer local_best_positions");
    ctx->local_best_values = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer local_best_values");
    ctx->best_position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_position");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
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
}

void FSA_cleanup_context(FSAContext *ctx, FSACLContext *cl_ctx) {
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->local_best_values) clReleaseMemObject(ctx->local_best_values);
    if (ctx->local_best_positions) clReleaseMemObject(ctx->local_best_positions);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

void FSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Firefly Swarm Algorithm optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("FSA.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open FSA.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: FSA.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read FSA.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: FSA.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    FSACLContext cl_ctx = {0};
    FSA_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    FSAContext ctx = {0};
    FSA_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;
    int run;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * pop_size : MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_pop_time;
        double update_pop_time;
        double initial_strategy_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;
    cl_mem best_fitness_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_fitness");

    float *best_scores = (float *)malloc(FSA_NUM_RUNS * sizeof(float));
    float **best_positions = (float **)malloc(FSA_NUM_RUNS * sizeof(float *));
    for (run = 0; run < FSA_NUM_RUNS; run++) {
        best_positions[run] = (float *)malloc(dim * sizeof(float));
    }

    for (run = 0; run < FSA_NUM_RUNS; run++) {
        printf("Run %d of %d\n", run + 1, FSA_NUM_RUNS);

        // Initialize population
	cl_event init_event;
	clSetKernelArg(cl_ctx.init_pop_kernel, 0, sizeof(cl_mem), &ctx.population);
	clSetKernelArg(cl_ctx.init_pop_kernel, 1, sizeof(cl_mem), &ctx.fitness);
	clSetKernelArg(cl_ctx.init_pop_kernel, 2, sizeof(cl_mem), &ctx.local_best_positions);
	clSetKernelArg(cl_ctx.init_pop_kernel, 3, sizeof(cl_mem), &ctx.local_best_values);
	clSetKernelArg(cl_ctx.init_pop_kernel, 4, sizeof(cl_mem), &ctx.bounds);
	clSetKernelArg(cl_ctx.init_pop_kernel, 5, sizeof(cl_mem), &ctx.best_position);
	clSetKernelArg(cl_ctx.init_pop_kernel, 6, sizeof(cl_mem), &best_fitness_buf);
	clSetKernelArg(cl_ctx.init_pop_kernel, 7, sizeof(cl_mem), &ctx.random_seeds); // Add missing seeds argument
	clSetKernelArg(cl_ctx.init_pop_kernel, 8, sizeof(cl_int), &dim);
	clSetKernelArg(cl_ctx.init_pop_kernel, 9, sizeof(cl_int), &pop_size);

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
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.local_best_values, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer local_best_values");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.local_best_positions, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer local_best_positions");
        free(positions);

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

        float best_fitness = fitness[0];
        int best_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (fitness[i] < best_fitness) {
                best_fitness = fitness[i];
                best_idx = i;
            }
        }
        ctx.best_fitness = best_fitness;

        positions = (float *)malloc(dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer best_pos");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer best_pos");
        err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &best_fitness, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer best_fitness");
        free(positions);
        free(fitness);

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

        // Main optimization loop
        while (func_count < max_evals && iteration < FSA_MAX_ITERATIONS) {
            iteration++;
            #if FSA_DEBUG
            printf("Run %d, Iteration %d\n", run + 1, iteration);
            #endif

            // Update population
            cl_event update_event;
            clSetKernelArg(cl_ctx.update_pop_kernel, 0, sizeof(cl_mem), &ctx.population);
            clSetKernelArg(cl_ctx.update_pop_kernel, 1, sizeof(cl_mem), &ctx.fitness);
            clSetKernelArg(cl_ctx.update_pop_kernel, 2, sizeof(cl_mem), &ctx.local_best_positions);
            clSetKernelArg(cl_ctx.update_pop_kernel, 3, sizeof(cl_mem), &ctx.local_best_values);
            clSetKernelArg(cl_ctx.update_pop_kernel, 4, sizeof(cl_mem), &ctx.best_position);
            clSetKernelArg(cl_ctx.update_pop_kernel, 5, sizeof(cl_mem), &best_fitness_buf);
            clSetKernelArg(cl_ctx.update_pop_kernel, 6, sizeof(cl_mem), &ctx.bounds);
            clSetKernelArg(cl_ctx.update_pop_kernel, 7, sizeof(cl_mem), &ctx.random_seeds);
            clSetKernelArg(cl_ctx.update_pop_kernel, 8, sizeof(cl_int), &dim);
            clSetKernelArg(cl_ctx.update_pop_kernel, 9, sizeof(cl_int), &pop_size);

            err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_pop_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &update_event);
            check_cl_error(err, "clEnqueueNDRangeKernel update_population");
            clFinish(cl_ctx.queue);

            double update_time = 0.0;
            if (update_event) {
                err = clGetEventProfilingInfo(update_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(update_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        update_time = (end - start) / 1e6;
                        prof.update_pop_time += update_time;
                    }
                }
                clReleaseEvent(update_event);
            }

            // Evaluate population on CPU
            positions = (float *)malloc(pop_size * dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer population");
            fitness = (float *)malloc(pop_size * sizeof(float));
            for (int i = 0; i < pop_size; i++) {
                double *pos = (double *)malloc(dim * sizeof(double));
                for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
                fitness[i] = (float)objective_function(pos);
                free(pos);
            }
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer fitness");
            func_count += pop_size;

            // Update local bests
            float *local_best_vals = (float *)malloc(pop_size * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.local_best_values, CL_TRUE, 0, pop_size * sizeof(cl_float), local_best_vals, 0, NULL, NULL);
            check_cl_error(err, "clEnqueueReadBuffer local_best_values");
            for (int i = 0; i < pop_size; i++) {
                if (fitness[i] <= local_best_vals[i]) {
                    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.local_best_values, CL_TRUE, i * sizeof(cl_float), sizeof(cl_float), &fitness[i], 0, NULL, NULL);
                    check_cl_error(err, "clEnqueueWriteBuffer local_best_values single");
                    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.local_best_positions, CL_TRUE, i * dim * sizeof(cl_float), dim * sizeof(cl_float), &positions[i * dim], 0, NULL, NULL);
                    check_cl_error(err, "clEnqueueWriteBuffer local_best_positions single");
                }
            }
            free(local_best_vals);

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

            // Update global best
            float new_best = fitness[0];
            int best_idx = 0;
            for (int i = 1; i < pop_size; i++) {
                if (fitness[i] < new_best) {
                    new_best = fitness[i];
                    best_idx = i;
                }
            }
            if (new_best < ctx.best_fitness) {
                ctx.best_fitness = new_best;
                positions = (float *)malloc(dim * sizeof(float));
                err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
                check_cl_error(err, "clEnqueueReadBuffer best_pos");
                err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
                check_cl_error(err, "clEnqueueWriteBuffer best_pos");
                err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &new_best, 0, NULL, NULL);
                check_cl_error(err, "clEnqueueWriteBuffer best_fitness");
                free(positions);

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
            }

            // Initial strategy update
            cl_event initial_strategy_event;
            clSetKernelArg(cl_ctx.initial_strategy_kernel, 0, sizeof(cl_mem), &ctx.population);
            clSetKernelArg(cl_ctx.initial_strategy_kernel, 1, sizeof(cl_mem), &ctx.fitness);
            clSetKernelArg(cl_ctx.initial_strategy_kernel, 2, sizeof(cl_mem), &ctx.local_best_positions);
            clSetKernelArg(cl_ctx.initial_strategy_kernel, 3, sizeof(cl_mem), &ctx.local_best_values);
            clSetKernelArg(cl_ctx.initial_strategy_kernel, 4, sizeof(cl_mem), &ctx.best_position);
            clSetKernelArg(cl_ctx.initial_strategy_kernel, 5, sizeof(cl_mem), &best_fitness_buf);
            clSetKernelArg(cl_ctx.initial_strategy_kernel, 6, sizeof(cl_mem), &ctx.bounds);
            clSetKernelArg(cl_ctx.initial_strategy_kernel, 7, sizeof(cl_mem), &ctx.random_seeds);
            clSetKernelArg(cl_ctx.initial_strategy_kernel, 8, sizeof(cl_int), &dim);
            clSetKernelArg(cl_ctx.initial_strategy_kernel, 9, sizeof(cl_int), &pop_size);

            err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.initial_strategy_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &initial_strategy_event);
            check_cl_error(err, "clEnqueueNDRangeKernel initial_strategy_update");
            clFinish(cl_ctx.queue);

            double initial_strategy_time = 0.0;
            if (initial_strategy_event) {
                err = clGetEventProfilingInfo(initial_strategy_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(initial_strategy_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        initial_strategy_time = (end - start) / 1e6;
                        prof.initial_strategy_time += initial_strategy_time;
                    }
                }
                clReleaseEvent(initial_strategy_event);
            }

            // Evaluate population on CPU
            positions = (float *)malloc(pop_size * dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer population");
            fitness = (float *)malloc(pop_size * sizeof(float));
            for (int i = 0; i < pop_size; i++) {
                double *pos = (double *)malloc(dim * sizeof(double));
                for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
                fitness[i] = (float)objective_function(pos);
                free(pos);
            }
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer fitness");
            func_count += pop_size;

            // Update local bests
            local_best_vals = (float *)malloc(pop_size * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.local_best_values, CL_TRUE, 0, pop_size * sizeof(cl_float), local_best_vals, 0, NULL, NULL);
            check_cl_error(err, "clEnqueueReadBuffer local_best_values");
            for (int i = 0; i < pop_size; i++) {
                if (fitness[i] <= local_best_vals[i]) {
                    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.local_best_values, CL_TRUE, i * sizeof(cl_float), sizeof(cl_float), &fitness[i], 0, NULL, NULL);
                    check_cl_error(err, "clEnqueueWriteBuffer local_best_values single");
                    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.local_best_positions, CL_TRUE, i * dim * sizeof(cl_float), dim * sizeof(cl_float), &positions[i * dim], 0, NULL, NULL);
                    check_cl_error(err, "clEnqueueWriteBuffer local_best_positions single");
                }
            }
            free(local_best_vals);

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

            // Update global best
            new_best = fitness[0];
            best_idx = 0;
            for (int i = 1; i < pop_size; i++) {
                if (fitness[i] < new_best) {
                    new_best = fitness[i];
                    best_idx = i;
                }
            }
            if (new_best < ctx.best_fitness) {
                ctx.best_fitness = new_best;
                positions = (float *)malloc(dim * sizeof(float));
                err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
                check_cl_error(err, "clEnqueueReadBuffer best_pos");
                err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
                check_cl_error(err, "clEnqueueWriteBuffer best_pos");
                err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &new_best, 0, NULL, NULL);
                check_cl_error(err, "clEnqueueWriteBuffer best_fitness");
                free(positions);

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
            }
            free(fitness);

            prof.count++;
            #if FSA_DEBUG
            printf("Run %d, Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", run + 1, iteration, func_count, ctx.best_fitness);
            printf("Profiling (ms): Update Population = %.3f, Initial Strategy = %.3f\n", update_time, initial_strategy_time);
            #endif
        }

        // Store best result for this run
        best_scores[run] = ctx.best_fitness;
        positions = (float *)malloc(dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer best_pos final");
        memcpy(best_positions[run], positions, dim * sizeof(float));
        free(positions);
    }

    // Find best result across all runs
    int best_run = 0;
    for (run = 1; run < FSA_NUM_RUNS; run++) {
        if (best_scores[run] < best_scores[best_run]) {
            best_run = run;
        }
    }
    ctx.best_fitness = best_scores[best_run];
    opt->best_solution.fitness = ctx.best_fitness;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = best_positions[best_run][i];
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_pop_time);
    printf("Update Population: %.3f\n", prof.update_pop_time);
    printf("Initial Strategy Update: %.3f\n", prof.initial_strategy_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Update Population: %.3f\n", prof.update_pop_time / prof.count);
        printf("Initial Strategy Update: %.3f\n", prof.initial_strategy_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    #if FSA_DEBUG
    printf("Best Score across all runs: %f\n", ctx.best_fitness);
    printf("Best Position: ");
    for (int j = 0; j < dim; j++) {
        printf("%f ", opt->best_solution.position[j]);
    }
    printf("\n");
    #endif

    // Cleanup
    for (run = 0; run < FSA_NUM_RUNS; run++) {
        free(best_positions[run]);
    }
    free(best_positions);
    free(best_scores);
    clReleaseMemObject(best_fitness_buf);
    FSA_cleanup_context(&ctx, &cl_ctx);
    FSA_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
