#include "MFO.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

// External utility functions from FirefA
extern void check_cl_error(cl_int err, const char *msg);
extern char *preprocess_kernel_source(const char *source);

void MFO_init_cl(MFOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->update_kernel = clCreateKernel(cl_ctx->program, "update_moths", &err);
    check_cl_error(err, "clCreateKernel update_moths");
    cl_ctx->sort_kernel = clCreateKernel(cl_ctx->program, "bitonic_sort", &err);
    check_cl_error(err, "clCreateKernel bitonic_sort");
    cl_ctx->update_flames_kernel = clCreateKernel(cl_ctx->program, "update_flames", &err);
    check_cl_error(err, "clCreateKernel update_flames");
}

void MFO_cleanup_cl(MFOCLContext *cl_ctx) {
    if (cl_ctx->update_flames_kernel) clReleaseKernel(cl_ctx->update_flames_kernel);
    if (cl_ctx->sort_kernel) clReleaseKernel(cl_ctx->sort_kernel);
    if (cl_ctx->update_kernel) clReleaseKernel(cl_ctx->update_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void MFO_init_context(MFOContext *ctx, Optimizer *opt, MFOCLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;

    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->population = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer population");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->best_flames = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_flames");
    ctx->best_flame_fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_flame_fitness");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->sorted_indices = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer sorted_indices");

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

void MFO_cleanup_context(MFOContext *ctx, MFOCLContext *cl_ctx) {
    if (ctx->sorted_indices) clReleaseMemObject(ctx->sorted_indices);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_flame_fitness) clReleaseMemObject(ctx->best_flame_fitness);
    if (ctx->best_flames) clReleaseMemObject(ctx->best_flames);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

void MFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Moth-Flame Optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("MFO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open MFO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: MFO.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read MFO.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: MFO.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    MFOCLContext cl_ctx = {0};
    MFO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    MFOContext ctx = {0};
    MFO_init_context(&ctx, opt, &cl_ctx);

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
        double update_time;
        double sort_time;
        double flames_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;
    cl_mem best_fitness_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_fitness");

    // Temporary buffer for combined population
    cl_mem temp_population = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, 2 * pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer temp_population");
    cl_mem temp_fitness = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, 2 * pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer temp_fitness");

    // Initialize population
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_pop_kernel, 0, sizeof(cl_mem), &ctx.population);
    clSetKernelArg(cl_ctx.init_pop_kernel, 1, sizeof(cl_mem), &ctx.fitness);
    clSetKernelArg(cl_ctx.init_pop_kernel, 2, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 4, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_pop_kernel, 5, sizeof(cl_int), &pop_size);

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

    // Initialize best flames
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_flames, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer best_flames");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_flame_fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer best_flame_fitness");
    free(positions);
    free(fitness);

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

        // Update flames on GPU
        cl_event flames_event;
        clSetKernelArg(cl_ctx.update_flames_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.update_flames_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.update_flames_kernel, 2, sizeof(cl_mem), &ctx.best_flames);
        clSetKernelArg(cl_ctx.update_flames_kernel, 3, sizeof(cl_mem), &ctx.best_flame_fitness);
        clSetKernelArg(cl_ctx.update_flames_kernel, 4, sizeof(cl_mem), &temp_population);
        clSetKernelArg(cl_ctx.update_flames_kernel, 5, sizeof(cl_mem), &temp_fitness);
        clSetKernelArg(cl_ctx.update_flames_kernel, 6, sizeof(cl_int), &pop_size);
        clSetKernelArg(cl_ctx.update_flames_kernel, 7, sizeof(cl_int), &dim);

        size_t flames_work_size = 2 * pop_size; // For combined population
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_flames_kernel, 1, NULL, &flames_work_size, NULL, 0, NULL, &flames_event);
        check_cl_error(err, "clEnqueueNDRangeKernel update_flames");
        clFinish(cl_ctx.queue);

        double flames_time = 0.0;
        if (flames_event) {
            err = clGetEventProfilingInfo(flames_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(flames_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    flames_time = (end - start) / 1e6;
                    prof.flames_time += flames_time;
                }
            }
            clReleaseEvent(flames_event);
        }

        // Update moth positions
        cl_event update_event;
        cl_float t = (cl_float)iteration / opt->max_iter;
        cl_float a = MFO_A_INITIAL - (MFO_A_INITIAL - MFO_A_FINAL) * (1.0f - exp(-3.0f * t));
        cl_int flame_no = (cl_int)round(pop_size - iteration * ((pop_size - 1.0) / opt->max_iter));

        clSetKernelArg(cl_ctx.update_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.update_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.update_kernel, 2, sizeof(cl_mem), &ctx.best_flames);
        clSetKernelArg(cl_ctx.update_kernel, 3, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.update_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.update_kernel, 5, sizeof(cl_float), &a);
        clSetKernelArg(cl_ctx.update_kernel, 6, sizeof(cl_int), &flame_no);
        clSetKernelArg(cl_ctx.update_kernel, 7, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.update_kernel, 8, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &update_event);
        check_cl_error(err, "clEnqueueNDRangeKernel update_moths");
        clFinish(cl_ctx.queue);

        double update_time = 0.0;
        if (update_event) {
            err = clGetEventProfilingInfo(update_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(update_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    update_time = (end - start) / 1e6;
                    prof.update_time += update_time;
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
        free(positions);
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
    float new_best = fitness[0];
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
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_flames, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
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

        err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &new_best, 0, NULL, &write_event);
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
    free(fitness);

    prof.count++;
    printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.best_fitness);
    printf("Profiling (ms): Update Flames = %.3f, Update Moths = %.3f\n", flames_time, update_time);
}

// Print profiling summary
printf("\nProfiling Summary (Total ms):\n");
printf("Initialize Population: %.3f\n", prof.init_pop_time);
printf("Update Flames: %.3f\n", prof.flames_time);
printf("Update Moths: %.3f\n", prof.update_time);
printf("Data Reads: %.3f\n", prof.read_time);
printf("Data Writes: %.3f\n", prof.write_time);
if (prof.count > 0) {
    printf("Average per Iteration (ms):\n");
    printf("Update Flames: %.3f\n", prof.flames_time / prof.count);
    printf("Update Moths: %.3f\n", prof.update_time / prof.count);
    printf("Data Reads: %.3f\n", prof.read_time / prof.count);
    printf("Data Writes: %.3f\n", prof.write_time / prof.count);
}

// Copy best solution to optimizer
float *best_pos = (float *)malloc(dim * sizeof(float));
err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_flames, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
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
clReleaseMemObject(temp_fitness);
clReleaseMemObject(temp_population);
clReleaseMemObject(best_fitness_buf);
MFO_cleanup_context(&ctx, &cl_ctx);
MFO_cleanup_cl(&cl_ctx);

printf("Optimization completed\n");
}
