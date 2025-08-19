#include "ARFO.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (placeholder, similar to EFO and FlyFO)
char *preprocess_kernel_source(const char *source);

void ARFO_init_cl(ARFOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->regrowth_kernel = clCreateKernel(cl_ctx->program, "regrowth_phase", &err);
    check_cl_error(err, "clCreateKernel regrowth");
    cl_ctx->branching_kernel = clCreateKernel(cl_ctx->program, "branching_phase", &err);
    check_cl_error(err, "clCreateKernel branching");
    cl_ctx->lateral_growth_kernel = clCreateKernel(cl_ctx->program, "lateral_growth_phase", &err);
    check_cl_error(err, "clCreateKernel lateral_growth");
    cl_ctx->elimination_kernel = clCreateKernel(cl_ctx->program, "elimination_phase", &err);
    check_cl_error(err, "clCreateKernel elimination");
    cl_ctx->replenish_kernel = clCreateKernel(cl_ctx->program, "replenish_phase", &err);
    check_cl_error(err, "clCreateKernel replenish");
}

void ARFO_cleanup_cl(ARFOCLContext *cl_ctx) {
    if (cl_ctx->replenish_kernel) clReleaseKernel(cl_ctx->replenish_kernel);
    if (cl_ctx->elimination_kernel) clReleaseKernel(cl_ctx->elimination_kernel);
    if (cl_ctx->lateral_growth_kernel) clReleaseKernel(cl_ctx->lateral_growth_kernel);
    if (cl_ctx->branching_kernel) clReleaseKernel(cl_ctx->branching_kernel);
    if (cl_ctx->regrowth_kernel) clReleaseKernel(cl_ctx->regrowth_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void ARFO_init_context(ARFOContext *ctx, Optimizer *opt, ARFOCLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;
    cl_int pop_size = opt->population_size;
    ctx->original_pop_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->population = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer population");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->auxin = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer auxin");
    ctx->auxin_sorted = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer auxin_sorted");
    ctx->topology = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * 4 * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer topology");
    ctx->new_roots = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * ARFO_MAX_BRANCHING * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer new_roots");
    ctx->fitness_indices = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(ARFOFitnessIndex), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness_indices");
    ctx->best_position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_position");
    ctx->new_root_count = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer new_root_count");
    ctx->population_size = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer population_size");

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

    // Initialize topology (Von Neumann)
    int *topology = (int *)malloc(pop_size * 4 * sizeof(int));
    int rows = (int)sqrt((double)pop_size);
    int cols = (pop_size + rows - 1) / rows;
    for (int i = 0; i < pop_size * 4; i++) topology[i] = -1;
    for (int i = 0; i < pop_size; i++) {
        int row = i / cols;
        int col = i % cols;
        if (col > 0) topology[i * 4 + 0] = i - 1; // Left
        if (col < cols - 1 && i + 1 < pop_size) topology[i * 4 + 1] = i + 1; // Right
        if (row > 0) topology[i * 4 + 2] = i - cols; // Up
        if (row < (pop_size + cols - 1) / cols - 1 && i + cols < pop_size) topology[i * 4 + 3] = i + cols; // Down
    }
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->topology, CL_TRUE, 0, pop_size * 4 * sizeof(cl_int), topology, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer topology");
    free(topology);

    // Initialize counters
    cl_int zero = 0;
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->new_root_count, CL_TRUE, 0, sizeof(cl_int), &zero, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer new_root_count");
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->population_size, CL_TRUE, 0, sizeof(cl_int), &pop_size, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer population_size");
}

void ARFO_cleanup_context(ARFOContext *ctx, ARFOCLContext *cl_ctx) {
    if (ctx->population_size) clReleaseMemObject(ctx->population_size);
    if (ctx->new_root_count) clReleaseMemObject(ctx->new_root_count);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->fitness_indices) clReleaseMemObject(ctx->fitness_indices);
    if (ctx->new_roots) clReleaseMemObject(ctx->new_roots);
    if (ctx->topology) clReleaseMemObject(ctx->topology);
    if (ctx->auxin_sorted) clReleaseMemObject(ctx->auxin_sorted);
    if (ctx->auxin) clReleaseMemObject(ctx->auxin);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

void ARFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Artificial Root Foraging Optimization...\n\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device ||
        !opt->population_buffer || !opt->fitness_buffer ||
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("ARFO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open ARFO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: ARFO.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read ARFO.cl completely: read %zu, expected %zu\n", read_size, source_size);
        free(source);
        exit(1);
    }
    source[source_size] = '\0';

    int valid_content = 0;
    for (size_t i = 0; i < source_size; i++) {
        if (!isspace((unsigned char)source[i]) && isprint((unsigned char)source[i])) {
            valid_content = 1;
            break;
        }
    }
    if (!valid_content) {
        fprintf(stderr, "Error: ARFO.cl contains only whitespace or invalid characters\n");
        free(source);
        exit(1);
    }

    // Initialize OpenCL context
    ARFOCLContext cl_ctx = {0};
    ARFO_init_cl(&cl_ctx, source, opt);
    free(source);

    // Initialize algorithm context
    ARFOContext ctx = {0};
    ARFO_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * pop_size : ARFO_MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_pop_time;
        double regrowth_time;
        double branching_time;
        double lateral_growth_time;
        double elimination_time;
        double replenish_time;
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

        // Regrowth phase
        cl_event regrowth_event;
        err = clSetKernelArg(cl_ctx.regrowth_kernel, 0, sizeof(cl_mem), &ctx.population);
        check_cl_error(err, "clSetKernelArg regrowth 0");
        err = clSetKernelArg(cl_ctx.regrowth_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        check_cl_error(err, "clSetKernelArg regrowth 1");
        err = clSetKernelArg(cl_ctx.regrowth_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        check_cl_error(err, "clSetKernelArg regrowth 2");
        err = clSetKernelArg(cl_ctx.regrowth_kernel, 3, sizeof(cl_mem), &ctx.auxin);
        check_cl_error(err, "clSetKernelArg regrowth 3");
        err = clSetKernelArg(cl_ctx.regrowth_kernel, 4, sizeof(cl_mem), &ctx.auxin_sorted);
        check_cl_error(err, "clSetKernelArg regrowth 4");
        err = clSetKernelArg(cl_ctx.regrowth_kernel, 5, sizeof(cl_mem), &ctx.topology);
        check_cl_error(err, "clSetKernelArg regrowth 5");
        err = clSetKernelArg(cl_ctx.regrowth_kernel, 6, sizeof(cl_mem), &ctx.random_seeds);
        check_cl_error(err, "clSetKernelArg regrowth 6");
        err = clSetKernelArg(cl_ctx.regrowth_kernel, 7, sizeof(cl_int), &dim);
        check_cl_error(err, "clSetKernelArg regrowth 7");
        err = clSetKernelArg(cl_ctx.regrowth_kernel, 8, sizeof(cl_int), &pop_size);
        check_cl_error(err, "clSetKernelArg regrowth 8");

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.regrowth_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &regrowth_event);
        check_cl_error(err, "clEnqueueNDRangeKernel regrowth");
        clFinish(cl_ctx.queue);

        double regrowth_time = 0.0;
        if (regrowth_event) {
            err = clGetEventProfilingInfo(regrowth_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(regrowth_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    regrowth_time = (end - start) / 1e6;
                    prof.regrowth_time += regrowth_time;
                }
            }
            clReleaseEvent(regrowth_event);
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
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer fitness");
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

        // Branching phase
        cl_event branching_event;
        cl_int new_root_count_host = 0;
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.new_root_count, CL_TRUE, 0, sizeof(cl_int), &new_root_count_host, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer new_root_count");
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

        err = clSetKernelArg(cl_ctx.branching_kernel, 0, sizeof(cl_mem), &ctx.population);
        check_cl_error(err, "clSetKernelArg branching 0");
        err = clSetKernelArg(cl_ctx.branching_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        check_cl_error(err, "clSetKernelArg branching 1");
        err = clSetKernelArg(cl_ctx.branching_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        check_cl_error(err, "clSetKernelArg branching 2");
        err = clSetKernelArg(cl_ctx.branching_kernel, 3, sizeof(cl_mem), &ctx.auxin);
        check_cl_error(err, "clSetKernelArg branching 3");
        err = clSetKernelArg(cl_ctx.branching_kernel, 4, sizeof(cl_mem), &ctx.new_roots);
        check_cl_error(err, "clSetKernelArg branching 4");
        err = clSetKernelArg(cl_ctx.branching_kernel, 5, sizeof(cl_mem), &ctx.fitness_indices);
        check_cl_error(err, "clSetKernelArg branching 5");
        err = clSetKernelArg(cl_ctx.branching_kernel, 6, sizeof(cl_mem), &ctx.random_seeds);
        check_cl_error(err, "clSetKernelArg branching 6");
        err = clSetKernelArg(cl_ctx.branching_kernel, 7, sizeof(cl_int), &dim);
        check_cl_error(err, "clSetKernelArg branching 7");
        err = clSetKernelArg(cl_ctx.branching_kernel, 8, sizeof(cl_int), &pop_size);
        check_cl_error(err, "clSetKernelArg branching 8");
        err = clSetKernelArg(cl_ctx.branching_kernel, 9, sizeof(cl_int), &iteration);
        check_cl_error(err, "clSetKernelArg branching 9");
        int max_iter = opt->max_iter > 0 ? opt->max_iter : ARFO_MAX_EVALS_DEFAULT / pop_size;
        err = clSetKernelArg(cl_ctx.branching_kernel, 10, sizeof(cl_int), &max_iter);
        check_cl_error(err, "clSetKernelArg branching 10");
        err = clSetKernelArg(cl_ctx.branching_kernel, 11, sizeof(cl_mem), &ctx.new_root_count);
        check_cl_error(err, "clSetKernelArg branching 11");

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.branching_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &branching_event);
        check_cl_error(err, "clEnqueueNDRangeKernel branching");
        clFinish(cl_ctx.queue);

        double branching_time = 0.0;
        if (branching_event) {
            err = clGetEventProfilingInfo(branching_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(branching_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    branching_time = (end - start) / 1e6;
                    prof.branching_time += branching_time;
                }
            }
            clReleaseEvent(branching_event);
        }

        // Evaluate new roots
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.new_root_count, CL_TRUE, 0, sizeof(cl_int), &new_root_count_host, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer new_root_count");
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
        if (new_root_count_host > 0) {
            float *new_roots = (float *)malloc(new_root_count_host * dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.new_roots, CL_TRUE, 0, new_root_count_host * dim * sizeof(cl_float), new_roots, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer new_roots");
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
            for (int i = 0; i < new_root_count_host; i++) {
                double *pos = (double *)malloc(dim * sizeof(double));
                for (int j = 0; j < dim; j++) pos[j] = new_roots[i * dim + j];
                float new_fitness = (float)objective_function(pos);
                free(pos);
                func_count++;
                if (new_fitness < ctx.best_fitness) {
                    ctx.best_fitness = new_fitness;
                    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), &new_roots[i * dim], 0, NULL, &write_event);
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
                    err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &new_fitness, 0, NULL, &write_event);
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
            }
            free(new_roots);
        }

        // Lateral growth phase
        cl_event lateral_growth_event;
        err = clSetKernelArg(cl_ctx.lateral_growth_kernel, 0, sizeof(cl_mem), &ctx.population);
        check_cl_error(err, "clSetKernelArg lateral_growth 0");
        err = clSetKernelArg(cl_ctx.lateral_growth_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        check_cl_error(err, "clSetKernelArg lateral_growth 1");
        err = clSetKernelArg(cl_ctx.lateral_growth_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        check_cl_error(err, "clSetKernelArg lateral_growth 2");
        err = clSetKernelArg(cl_ctx.lateral_growth_kernel, 3, sizeof(cl_mem), &ctx.auxin);
        check_cl_error(err, "clSetKernelArg lateral_growth 3");
        err = clSetKernelArg(cl_ctx.lateral_growth_kernel, 4, sizeof(cl_mem), &ctx.auxin_sorted);
        check_cl_error(err, "clSetKernelArg lateral_growth 4");
        err = clSetKernelArg(cl_ctx.lateral_growth_kernel, 5, sizeof(cl_mem), &ctx.new_roots);
        check_cl_error(err, "clSetKernelArg lateral_growth 5");
        err = clSetKernelArg(cl_ctx.lateral_growth_kernel, 6, sizeof(cl_mem), &ctx.random_seeds);
        check_cl_error(err, "clSetKernelArg lateral_growth 6");
        err = clSetKernelArg(cl_ctx.lateral_growth_kernel, 7, sizeof(cl_int), &dim);
        check_cl_error(err, "clSetKernelArg lateral_growth 7");
        err = clSetKernelArg(cl_ctx.lateral_growth_kernel, 8, sizeof(cl_int), &pop_size);
        check_cl_error(err, "clSetKernelArg lateral_growth 8");

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.lateral_growth_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &lateral_growth_event);
        check_cl_error(err, "clEnqueueNDRangeKernel lateral_growth");
        clFinish(cl_ctx.queue);

        double lateral_growth_time = 0.0;
        if (lateral_growth_event) {
            err = clGetEventProfilingInfo(lateral_growth_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(lateral_growth_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    lateral_growth_time = (end - start) / 1e6;
                    prof.lateral_growth_time += lateral_growth_time;
                }
            }
            clReleaseEvent(lateral_growth_event);
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
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer fitness");
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

        // Elimination phase
        cl_event elimination_event;
        cl_int new_pop_size_host = 0;
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.population_size, CL_TRUE, 0, sizeof(cl_int), &new_pop_size_host, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer population_size");
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

        err = clSetKernelArg(cl_ctx.elimination_kernel, 0, sizeof(cl_mem), &ctx.population);
        check_cl_error(err, "clSetKernelArg elimination 0");
        err = clSetKernelArg(cl_ctx.elimination_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        check_cl_error(err, "clSetKernelArg elimination 1");
        err = clSetKernelArg(cl_ctx.elimination_kernel, 2, sizeof(cl_mem), &ctx.auxin);
        check_cl_error(err, "clSetKernelArg elimination 2");
        err = clSetKernelArg(cl_ctx.elimination_kernel, 3, sizeof(cl_mem), &ctx.auxin_sorted);
        check_cl_error(err, "clSetKernelArg elimination 3");
        err = clSetKernelArg(cl_ctx.elimination_kernel, 4, sizeof(cl_int), &dim);
        check_cl_error(err, "clSetKernelArg elimination 4");
        err = clSetKernelArg(cl_ctx.elimination_kernel, 5, sizeof(cl_int), &pop_size);
        check_cl_error(err, "clSetKernelArg elimination 5");
        err = clSetKernelArg(cl_ctx.elimination_kernel, 6, sizeof(cl_mem), &ctx.population_size);
        check_cl_error(err, "clSetKernelArg elimination 6");

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.elimination_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &elimination_event);
        check_cl_error(err, "clEnqueueNDRangeKernel elimination");
        clFinish(cl_ctx.queue);

        double elimination_time = 0.0;
        if (elimination_event) {
            err = clGetEventProfilingInfo(elimination_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(elimination_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    elimination_time = (end - start) / 1e6;
                    prof.elimination_time += elimination_time;
                }
            }
            clReleaseEvent(elimination_event);
        }

        // Update population size
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population_size, CL_TRUE, 0, sizeof(cl_int), &new_pop_size_host, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population_size");
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
        pop_size = new_pop_size_host;

        // Replenish phase
        cl_event replenish_event;
        err = clSetKernelArg(cl_ctx.replenish_kernel, 0, sizeof(cl_mem), &ctx.population);
        check_cl_error(err, "clSetKernelArg replenish 0");
        err = clSetKernelArg(cl_ctx.replenish_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        check_cl_error(err, "clSetKernelArg replenish 1");
        err = clSetKernelArg(cl_ctx.replenish_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        check_cl_error(err, "clSetKernelArg replenish 2");
        err = clSetKernelArg(cl_ctx.replenish_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
        check_cl_error(err, "clSetKernelArg replenish 3");
        err = clSetKernelArg(cl_ctx.replenish_kernel, 4, sizeof(cl_int), &dim);
        check_cl_error(err, "clSetKernelArg replenish 4");
        err = clSetKernelArg(cl_ctx.replenish_kernel, 5, sizeof(cl_mem), &ctx.population_size);
        check_cl_error(err, "clSetKernelArg replenish 5");
        err = clSetKernelArg(cl_ctx.replenish_kernel, 6, sizeof(cl_int), &ctx.original_pop_size);
        check_cl_error(err, "clSetKernelArg replenish 6");

        global_work_size = ctx.original_pop_size; // Ensure enough work-items to replenish
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.replenish_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &replenish_event);
        check_cl_error(err, "clEnqueueNDRangeKernel replenish");
        clFinish(cl_ctx.queue);

        double replenish_time = 0.0;
        if (replenish_event) {
            err = clGetEventProfilingInfo(replenish_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(replenish_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    replenish_time = (end - start) / 1e6;
                    prof.replenish_time += replenish_time;
                }
            }
            clReleaseEvent(replenish_event);
        }

        // Evaluate replenished population
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population_size, CL_TRUE, 0, sizeof(cl_int), &pop_size, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population_size");
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

        positions = (float *)realloc(positions, pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population");
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

        // Update topology
        int *topology = (int *)malloc(pop_size * 4 * sizeof(int));
        int rows = (int)sqrt((double)pop_size);
        int cols = (pop_size + rows - 1) / rows;
        for (int i = 0; i < pop_size * 4; i++) topology[i] = -1;
        for (int i = 0; i < pop_size; i++) {
            int row = i / cols;
            int col = i % cols;
            if (col > 0) topology[i * 4 + 0] = i - 1;  // Left
            if (col < cols - 1 && i + 1 < pop_size) topology[i * 4 + 1] = i + 1;  // Right
            if (row > 0) topology[i * 4 + 2] = i - cols;  // Up
            if (row < (pop_size + cols - 1) / cols - 1 && i + cols < pop_size) topology[i * 4 + 3] = i + cols;  // Down
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.topology, CL_TRUE, 0, pop_size * 4 * sizeof(cl_int), topology, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer topology");
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
        free(topology);

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.best_fitness);
        printf("Profiling (ms): Regrowth = %.3f, Branching = %.3f, Lateral Growth = %.3f, Elimination = %.3f, Replenish = %.3f\n",
               regrowth_time, branching_time, lateral_growth_time, elimination_time, replenish_time);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_pop_time);
    printf("Regrowth Phase: %.3f\n", prof.regrowth_time);
    printf("Branching Phase: %.3f\n", prof.branching_time);
    printf("Lateral Growth Phase: %.3f\n", prof.lateral_growth_time);
    printf("Elimination Phase: %.3f\n", prof.elimination_time);
    printf("Replenish Phase: %.3f\n", prof.replenish_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("\nAverage per Iteration (ms):\n");
        printf("Regrowth Phase: %.3f\n", prof.regrowth_time / prof.count);
        printf("Branching Phase: %.3f\n", prof.branching_time / prof.count);
        printf("Lateral Growth Phase: %.3f\n", prof.lateral_growth_time / prof.count);
        printf("Elimination Phase: %.3f\n", prof.elimination_time / prof.count);
        printf("Replenish Phase: %.3f\n", prof.replenish_time / prof.count);
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

    // Cleanup
    clReleaseMemObject(best_fitness_buf);
    ARFO_cleanup_context(&ctx, &cl_ctx);
    ARFO_cleanup_cl(&cl_ctx);

    printf("\nOptimization completed.\n");
}
