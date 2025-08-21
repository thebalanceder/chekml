#include "IWD.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source to fix rand_float address space
char *preprocess_kernel_source(const char *source);

void IWD_init_cl(IWDCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
            fprintf(stderr, "Error building program: %d\nDetailed Build Log:\n%s\n", err, log);
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
    cl_ctx->move_drop_kernel = clCreateKernel(cl_ctx->program, "move_water_drop", &err);
    check_cl_error(err, "clCreateKernel move_water_drop");
    cl_ctx->update_soil_kernel = clCreateKernel(cl_ctx->program, "update_soil", &err);
    check_cl_error(err, "clCreateKernel update_soil");
}

void IWD_cleanup_cl(IWDCLContext *cl_ctx) {
    if (cl_ctx->update_soil_kernel) clReleaseKernel(cl_ctx->update_soil_kernel);
    if (cl_ctx->move_drop_kernel) clReleaseKernel(cl_ctx->move_drop_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void IWD_init_context(IWDContext *ctx, Optimizer *opt, IWDCLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;

    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->population = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer population");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->best_position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_position");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->soil = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer soil");
    ctx->hud = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer hud");
    ctx->velocities = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer velocities");

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

void IWD_cleanup_context(IWDContext *ctx, IWDCLContext *cl_ctx) {
    if (ctx->velocities) clReleaseMemObject(ctx->velocities);
    if (ctx->hud) clReleaseMemObject(ctx->hud);
    if (ctx->soil) clReleaseMemObject(ctx->soil);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

void IWD_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Intelligent Water Drops optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("IWD.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open IWD.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: IWD.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read IWD.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: IWD.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    IWDCLContext cl_ctx = {0};
    IWD_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    IWDContext ctx = {0};
    IWD_init_context(&ctx, opt, &cl_ctx);

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
        double move_drop_time;
        double update_soil_time;
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
    clSetKernelArg(cl_ctx.init_pop_kernel, 0, sizeof(cl_mem), &ctx.population);
    clSetKernelArg(cl_ctx.init_pop_kernel, 1, sizeof(cl_mem), &ctx.fitness);
    clSetKernelArg(cl_ctx.init_pop_kernel, 2, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 3, sizeof(cl_mem), &ctx.best_position);
    clSetKernelArg(cl_ctx.init_pop_kernel, 4, sizeof(cl_mem), &best_fitness_buf);
    clSetKernelArg(cl_ctx.init_pop_kernel, 5, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 6, sizeof(cl_mem), &ctx.soil);
    clSetKernelArg(cl_ctx.init_pop_kernel, 7, sizeof(cl_mem), &ctx.hud);
    clSetKernelArg(cl_ctx.init_pop_kernel, 8, sizeof(cl_mem), &ctx.velocities);
    clSetKernelArg(cl_ctx.init_pop_kernel, 9, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_pop_kernel, 10, sizeof(cl_int), &pop_size);

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
                prof.init_pop_time += (end - start) / 1e6;
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
    free(fitness);

    positions = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer best_pos");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer best_pos");
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
    cl_mem visited_flags = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_char), NULL, &err);
    check_cl_error(err, "clCreateBuffer visited_flags");
    int *visited = (int *)malloc(pop_size * pop_size * sizeof(int)); // Allow multiple paths
    int *visited_counts = (int *)malloc(pop_size * sizeof(int));
    float *soil_amounts = (float *)malloc(pop_size * sizeof(float));

    while (func_count < max_evals) {
        iteration++;
        printf("Iteration %d\n", iteration);

        // Reset visited flags and counts
        char *flags = (char *)calloc(pop_size, sizeof(char));
        for (int i = 0; i < pop_size; i++) {
            visited_counts[i] = 1;
            visited[i * pop_size] = i;
            soil_amounts[i] = 0.0f;
            flags[i] = 1; // Mark starting node as visited
        }

        // Move water drops until all nodes are visited
        int all_visited = 0;
        while (!all_visited) {
            all_visited = 1;
            for (int i = 0; i < pop_size; i++) {
                if (visited_counts[i] < pop_size) {
                    all_visited = 0;
                    // Reset flags for current water drop
                    for (int j = 0; j < pop_size; j++) {
                        flags[j] = (visited[i * pop_size + j] != 0);
                    }
                    flags[i] = 1; // Ensure current node is marked
                    err = clEnqueueWriteBuffer(cl_ctx.queue, visited_flags, CL_TRUE, 0, pop_size * sizeof(cl_char), flags, 0, NULL, NULL);
                    check_cl_error(err, "clEnqueueWriteBuffer visited_flags");

                    cl_event move_event;
                    clSetKernelArg(cl_ctx.move_drop_kernel, 0, sizeof(cl_mem), &ctx.population);
                    clSetKernelArg(cl_ctx.move_drop_kernel, 1, sizeof(cl_mem), &ctx.fitness);
                    clSetKernelArg(cl_ctx.move_drop_kernel, 2, sizeof(cl_mem), &ctx.bounds);
                    clSetKernelArg(cl_ctx.move_drop_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
                    clSetKernelArg(cl_ctx.move_drop_kernel, 4, sizeof(cl_mem), &ctx.soil);
                    clSetKernelArg(cl_ctx.move_drop_kernel, 5, sizeof(cl_mem), &ctx.hud);
                    clSetKernelArg(cl_ctx.move_drop_kernel, 6, sizeof(cl_mem), &ctx.velocities);
                    clSetKernelArg(cl_ctx.move_drop_kernel, 7, sizeof(cl_mem), &visited_flags);
                    clSetKernelArg(cl_ctx.move_drop_kernel, 8, sizeof(cl_int), &dim);
                    clSetKernelArg(cl_ctx.move_drop_kernel, 9, sizeof(cl_int), &pop_size);

                    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.move_drop_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &move_event);
                    check_cl_error(err, "clEnqueueNDRangeKernel move_water_drop");
                    clFinish(cl_ctx.queue);

                    if (move_event) {
                        err = clGetEventProfilingInfo(move_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                        if (err == CL_SUCCESS) {
                            err = clGetEventProfilingInfo(move_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                            if (err == CL_SUCCESS) {
                                prof.move_drop_time += (end - start) / 1e6;
                            }
                        }
                        clReleaseEvent(move_event);
                    }

                    // Read back visited flags and update counts
                    err = clEnqueueReadBuffer(cl_ctx.queue, visited_flags, CL_TRUE, 0, pop_size * sizeof(cl_char), flags, 0, NULL, &read_event);
                    check_cl_error(err, "clEnqueueReadBuffer visited_flags");
                    for (int j = 0; j < pop_size; j++) {
                        if (flags[j] && j != i) {
                            int found = 0;
                            for (int k = 0; k < visited_counts[i]; k++) {
                                if (visited[i * pop_size + k] == j) {
                                    found = 1;
                                    break;
                                }
                            }
                            if (!found) {
                                visited[i * pop_size + visited_counts[i]] = j;
                                visited_counts[i]++;
                                soil_amounts[i] += 1.0f; // Increment soil amount per move
                            }
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
                }
            }
        }

        free(flags);

        // Evaluate population on CPU
        positions = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population");
        fitness = (float *)malloc(pop_size * sizeof(float));
        double *qualities = (double *)malloc(pop_size * sizeof(double));
        int max_quality_idx = 0;
        double max_quality = -INFINITY;
        int best_visited_count = 0;

        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            fitness[i] = (float)objective_function(pos);
            qualities[i] = (fitness[i] != 0.0) ? 1.0 / fitness[i] : INFINITY;
            if (qualities[i] > max_quality) {
                max_quality = qualities[i];
                max_quality_idx = i;
                best_visited_count = visited_counts[i];
            }
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer fitness");
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

        // Update soil for best water drop
        cl_mem visited_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, best_visited_count * sizeof(cl_int), NULL, &err);
        check_cl_error(err, "clCreateBuffer visited");
        err = clEnqueueWriteBuffer(cl_ctx.queue, visited_buf, CL_TRUE, 0, best_visited_count * sizeof(cl_int), &visited[max_quality_idx * pop_size], 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer visited");

        cl_event update_soil_event;
        clSetKernelArg(cl_ctx.update_soil_kernel, 0, sizeof(cl_mem), &ctx.soil);
        clSetKernelArg(cl_ctx.update_soil_kernel, 1, sizeof(cl_mem), &visited_buf);
        clSetKernelArg(cl_ctx.update_soil_kernel, 2, sizeof(cl_int), &best_visited_count);
        clSetKernelArg(cl_ctx.update_soil_kernel, 3, sizeof(cl_float), &soil_amounts[max_quality_idx]);
        clSetKernelArg(cl_ctx.update_soil_kernel, 4, sizeof(cl_int), &pop_size);

        size_t soil_work_size = best_visited_count - 1;
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_soil_kernel, 1, NULL, &soil_work_size, NULL, 0, NULL, &update_soil_event);
        check_cl_error(err, "clEnqueueNDRangeKernel update_soil");
        clFinish(cl_ctx.queue);

        if (update_soil_event) {
            err = clGetEventProfilingInfo(update_soil_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(update_soil_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.update_soil_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(update_soil_event);
        }
        clReleaseMemObject(visited_buf);

        // Update global best
        if (fitness[max_quality_idx] < ctx.best_fitness) {
            ctx.best_fitness = fitness[max_quality_idx];
            positions = (float *)malloc(dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, max_quality_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer best_pos");
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer best_pos");
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

            err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &ctx.best_fitness, 0, NULL, &write_event);
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

        // Reinitialize HUD
        float *all_pos = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), all_pos, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population for HUD");
        float *new_hud = (float *)malloc(pop_size * pop_size * sizeof(float));
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < pop_size; j++) {
                if (i == j) {
                    new_hud[i * pop_size + j] = 0.0f;
                } else {
                    float dist = 0.0f;
                    for (int k = 0; k < dim; k++) {
                        float diff = all_pos[i * dim + k] - all_pos[j * dim + k];
                        dist += diff * diff;
                    }
                    new_hud[i * pop_size + j] = sqrt(dist);
                }
            }
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.hud, CL_TRUE, 0, pop_size * pop_size * sizeof(cl_float), new_hud, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer hud");
        free(all_pos);
        free(new_hud);

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
        free(fitness);
        free(qualities);

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.best_fitness);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_pop_time);
    printf("Move Water Drops: %.3f\n", prof.move_drop_time);
    printf("Update Soil: %.3f\n", prof.update_soil_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Move Water Drops: %.3f\n", prof.move_drop_time / prof.count);
        printf("Update Soil: %.3f\n", prof.update_soil_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Copy best solution to optimizer
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
    clReleaseMemObject(visited_flags);
    free(visited);
    free(visited_counts);
    free(soil_amounts);
    clReleaseMemObject(best_fitness_buf);
    IWD_cleanup_context(&ctx, &cl_ctx);
    IWD_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
