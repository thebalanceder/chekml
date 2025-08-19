#include "CSS.h"
#include <string.h>
#include <time.h>
#include <float.h>

// Fast Xorshift random number generator state
static uint32_t xorshift_state = 1;

// Initialize Xorshift seed
void init_xorshift_css(uint32_t seed) {
    xorshift_state = seed ? seed : (uint32_t)time(NULL);
}

// Fast Xorshift random number generator
static inline uint32_t xorshift32_css() {
    uint32_t x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

// Generate random double between min and max
static inline double rand_double_css(double min, double max) {
    double r = (double)xorshift32_css() / UINT32_MAX;
    return min + r * (max - min);
}

// Error checking utility
static void check_cl_error(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error (%d): %s\n", err, msg);
        exit(1);
    }
}

// Preprocess kernel source (stub)
static char *preprocess_kernel_source(const char *source) {
    char *processed = (char *)malloc(strlen(source) + 1);
    if (!processed) {
        fprintf(stderr, "Memory allocation failed for kernel source\n");
        exit(1);
    }
    strcpy(processed, source);
    return processed;
}

void CSS_init_cl(CSSCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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

    const char *build_options = "-cl-std=CL1.1 -cl-mad-enable";
    err = clBuildProgram(cl_ctx->program, 1, &cl_ctx->device, build_options, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(cl_ctx->program, cl_ctx->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size + 1);
        if (log) {
            clGetProgramBuildInfo(cl_ctx->program, cl_ctx->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            log[log_size] = '\0';
            fprintf(stderr, "Error building program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(cl_ctx->program);
        free(processed_source);
        exit(1);
    }
    free(processed_source);

    // Create kernels
    cl_ctx->init_particles_kernel = clCreateKernel(cl_ctx->program, "init_particles", &err);
    check_cl_error(err, "clCreateKernel init_particles");
    cl_ctx->calc_forces_kernel = clCreateKernel(cl_ctx->program, "calc_forces", &err);
    check_cl_error(err, "clCreateKernel calc_forces");
    cl_ctx->update_positions_kernel = clCreateKernel(cl_ctx->program, "update_positions", &err);
    check_cl_error(err, "clCreateKernel update_positions");
    cl_ctx->update_cm_kernel = clCreateKernel(cl_ctx->program, "update_cm", &err);
    check_cl_error(err, "clCreateKernel update_cm");
    cl_ctx->find_min_max_kernel = clCreateKernel(cl_ctx->program, "find_min_max", &err);
    check_cl_error(err, "clCreateKernel find_min_max");
}

void CSS_cleanup_cl(CSSCLContext *cl_ctx) {
    if (cl_ctx->find_min_max_kernel) clReleaseKernel(cl_ctx->find_min_max_kernel);
    if (cl_ctx->update_cm_kernel) clReleaseKernel(cl_ctx->update_cm_kernel);
    if (cl_ctx->update_positions_kernel) clReleaseKernel(cl_ctx->update_positions_kernel);
    if (cl_ctx->calc_forces_kernel) clReleaseKernel(cl_ctx->calc_forces_kernel);
    if (cl_ctx->init_particles_kernel) clReleaseKernel(cl_ctx->init_particles_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void CSS_init_context(CSSContext *ctx, Optimizer *opt, CSSCLContext *cl_ctx) {
    cl_int err;
    ctx->best_cost = INFINITY;

    int population_size = opt->population_size;
    int dim = opt->dim;
    int cm_size = (int)(CSS_CM_SIZE_RATIO * population_size);

    // Allocate OpenCL buffers
    ctx->positions = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer positions");
    ctx->costs = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer costs");
    ctx->forces = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer forces");
    ctx->velocities = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer velocities");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->cm_positions = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, cm_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer cm_positions");
    ctx->cm_costs = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, cm_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer cm_costs");
    ctx->min_max_indices = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, 2 * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer min_max_indices");

    // Initialize random seeds
    cl_uint *seeds = (cl_uint *)malloc(population_size * sizeof(cl_uint));
    init_xorshift_css((uint32_t)time(NULL));
    for (int i = 0; i < population_size; i++) seeds[i] = xorshift32_css();
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->random_seeds, CL_TRUE, 0, population_size * sizeof(cl_uint), seeds, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer seeds");
    free(seeds);

    // Initialize bounds
    float *bounds_float = (float *)malloc(dim * 2 * sizeof(float));
    for (int i = 0; i < dim * 2; i++) bounds_float[i] = (float)opt->bounds[i];
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->bounds, CL_TRUE, 0, dim * 2 * sizeof(cl_float), bounds_float, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer bounds");
    free(bounds_float);

    // Initialize velocities to zero
    float *zero_velocities = (float *)calloc(population_size * dim, sizeof(float));
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->velocities, CL_TRUE, 0, population_size * dim * sizeof(cl_float), zero_velocities, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer velocities");
    free(zero_velocities);
}

void CSS_cleanup_context(CSSContext *ctx, CSSCLContext *cl_ctx) {
    if (ctx->min_max_indices) clReleaseMemObject(ctx->min_max_indices);
    if (ctx->cm_costs) clReleaseMemObject(ctx->cm_costs);
    if (ctx->cm_positions) clReleaseMemObject(ctx->cm_positions);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->velocities) clReleaseMemObject(ctx->velocities);
    if (ctx->forces) clReleaseMemObject(ctx->forces);
    if (ctx->costs) clReleaseMemObject(ctx->costs);
    if (ctx->positions) clReleaseMemObject(ctx->positions);
}

void CSS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU CSS optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("CSS.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open CSS.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    rewind(fp);
    char *source_str = (char *)malloc(source_size + 1);
    if (!source_str) {
        fprintf(stderr, "Memory allocation failed for kernel source\n");
        fclose(fp);
        exit(1);
    }
    source_size = fread(source_str, 1, source_size, fp);
    fclose(fp);
    source_str[source_size] = '\0';

    // Initialize OpenCL context
    CSSCLContext cl_ctx = {0};
    CSS_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    CSSContext ctx = {0};
    CSS_init_context(&ctx, opt, &cl_ctx);

    int population_size = opt->population_size;
    int dim = opt->dim;
    int cm_size = (int)(CSS_CM_SIZE_RATIO * population_size);
    int func_count = 0;
    int iteration = 0;
    int stagnation_count = 0;
    float prev_best_cost = INFINITY;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * population_size : CSS_MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_time;
        double forces_time;
        double update_pos_time;
        double update_cm_time;
        double min_max_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;

    // Initialize particles
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_particles_kernel, 0, sizeof(cl_mem), &ctx.positions);
    clSetKernelArg(cl_ctx.init_particles_kernel, 1, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_particles_kernel, 2, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_particles_kernel, 3, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_particles_kernel, 4, sizeof(cl_int), &population_size);
    size_t global_work_size = population_size;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.init_particles_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
    check_cl_error(err, "clEnqueueNDRangeKernel init_particles");
    clFinish(cl_ctx.queue);

    cl_ulong start, end;
    if (init_event) {
        clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        prof.init_time += (end - start) / 1e6;
        clReleaseEvent(init_event);
    }

    // Evaluate initial population on CPU
    cl_event read_event, write_event;
    float *positions = (float *)malloc(population_size * dim * sizeof(float));
    float *costs = (float *)malloc(population_size * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer positions");
    for (int i = 0; i < population_size; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = (double)positions[i * dim + j];
        costs[i] = (float)objective_function(pos);
        costs[i] = isnan(costs[i]) || isinf(costs[i]) ? INFINITY : costs[i];
        if (costs[i] < ctx.best_cost) {
            ctx.best_cost = costs[i];
            opt->best_solution.fitness = ctx.best_cost;
            for (int j = 0; j < dim; j++) opt->best_solution.position[j] = pos[j];
        }
        free(pos);
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, population_size * sizeof(cl_float), costs, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer costs");

    if (read_event) {
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        prof.read_time += (end - start) / 1e6;
        clReleaseEvent(read_event);
    }
    if (write_event) {
        clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        prof.write_time += (end - start) / 1e6;
        clReleaseEvent(write_event);
    }

    func_count += population_size;

    // Initialize charged memory with initial population
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.cm_positions, CL_TRUE, 0, cm_size * dim * sizeof(cl_float), positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer cm_positions");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.cm_costs, CL_TRUE, 0, cm_size * sizeof(cl_float), costs, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer cm_costs");

    // Main optimization loop
    while (func_count < max_evals) {
        iteration++;

        // Find min/max indices
        cl_event min_max_event;
        clSetKernelArg(cl_ctx.find_min_max_kernel, 0, sizeof(cl_mem), &ctx.costs);
        clSetKernelArg(cl_ctx.find_min_max_kernel, 1, sizeof(cl_mem), &ctx.min_max_indices);
        clSetKernelArg(cl_ctx.find_min_max_kernel, 2, sizeof(cl_int), &population_size);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.find_min_max_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &min_max_event);
        check_cl_error(err, "clEnqueueNDRangeKernel find_min_max");
        clFinish(cl_ctx.queue);

        if (min_max_event) {
            clGetEventProfilingInfo(min_max_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(min_max_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            prof.min_max_time += (end - start) / 1e6;
            clReleaseEvent(min_max_event);
        }

        // Calculate forces
        cl_event forces_event;
        clSetKernelArg(cl_ctx.calc_forces_kernel, 0, sizeof(cl_mem), &ctx.positions);
        clSetKernelArg(cl_ctx.calc_forces_kernel, 1, sizeof(cl_mem), &ctx.costs);
        clSetKernelArg(cl_ctx.calc_forces_kernel, 2, sizeof(cl_mem), &ctx.forces);
        clSetKernelArg(cl_ctx.calc_forces_kernel, 3, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.calc_forces_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.calc_forces_kernel, 5, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.calc_forces_kernel, 6, sizeof(cl_int), &population_size);
        clSetKernelArg(cl_ctx.calc_forces_kernel, 7, sizeof(cl_float), &ctx.best_cost);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.calc_forces_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &forces_event);
        check_cl_error(err, "clEnqueueNDRangeKernel calc_forces");
        clFinish(cl_ctx.queue);

        if (forces_event) {
            clGetEventProfilingInfo(forces_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(forces_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            prof.forces_time += (end - start) / 1e6;
            clReleaseEvent(forces_event);
        }

        // Update positions
        cl_event update_pos_event;
        clSetKernelArg(cl_ctx.update_positions_kernel, 0, sizeof(cl_mem), &ctx.positions);
        clSetKernelArg(cl_ctx.update_positions_kernel, 1, sizeof(cl_mem), &ctx.forces);
        clSetKernelArg(cl_ctx.update_positions_kernel, 2, sizeof(cl_mem), &ctx.velocities);
        clSetKernelArg(cl_ctx.update_positions_kernel, 3, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.update_positions_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.update_positions_kernel, 5, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.update_positions_kernel, 6, sizeof(cl_int), &population_size);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_positions_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &update_pos_event);
        check_cl_error(err, "clEnqueueNDRangeKernel update_positions");
        clFinish(cl_ctx.queue);

        if (update_pos_event) {
            clGetEventProfilingInfo(update_pos_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(update_pos_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            prof.update_pos_time += (end - start) / 1e6;
            clReleaseEvent(update_pos_event);
        }

        // Evaluate updated population
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer positions");
        for (int i = 0; i < population_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = (double)positions[i * dim + j];
            costs[i] = (float)objective_function(pos);
            costs[i] = isnan(costs[i]) || isinf(costs[i]) ? INFINITY : costs[i];
            if (costs[i] < ctx.best_cost) {
                ctx.best_cost = costs[i];
                opt->best_solution.fitness = ctx.best_cost;
                for (int j = 0; j < dim; j++) opt->best_solution.position[j] = pos[j];
            }
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, population_size * sizeof(cl_float), costs, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer costs");
        func_count += population_size;

        if (read_event) {
            clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            prof.read_time += (end - start) / 1e6;
            clReleaseEvent(read_event);
        }
        if (write_event) {
            clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            prof.write_time += (end - start) / 1e6;
            clReleaseEvent(write_event);
        }

        // Update charged memory
        cl_event update_cm_event;
        clSetKernelArg(cl_ctx.update_cm_kernel, 0, sizeof(cl_mem), &ctx.positions);
        clSetKernelArg(cl_ctx.update_cm_kernel, 1, sizeof(cl_mem), &ctx.costs);
        clSetKernelArg(cl_ctx.update_cm_kernel, 2, sizeof(cl_mem), &ctx.cm_positions);
        clSetKernelArg(cl_ctx.update_cm_kernel, 3, sizeof(cl_mem), &ctx.cm_costs);
        clSetKernelArg(cl_ctx.update_cm_kernel, 4, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.update_cm_kernel, 5, sizeof(cl_int), &population_size);
        clSetKernelArg(cl_ctx.update_cm_kernel, 6, sizeof(cl_int), &cm_size);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_cm_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &update_cm_event);
        check_cl_error(err, "clEnqueueNDRangeKernel update_cm");
        clFinish(cl_ctx.queue);

        if (update_cm_event) {
            clGetEventProfilingInfo(update_cm_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(update_cm_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            prof.update_cm_time += (end - start) / 1e6;
            clReleaseEvent(update_cm_event);
        }

        // Convergence check
        if (fabs(prev_best_cost - ctx.best_cost) < CSS_CONVERGENCE_TOL) {
            stagnation_count++;
            if (stagnation_count >= CSS_STAGNATION_THRESHOLD) {
                printf("Convergence reached, stopping early\n");
                break;
            }
        } else {
            stagnation_count = 0;
        }
        prev_best_cost = ctx.best_cost;

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Cost = %f\n", iteration, func_count, ctx.best_cost);
    }

    // Validate final best solution
    double *best_pos = (double *)malloc(dim * sizeof(double));
    for (int i = 0; i < dim; i++) best_pos[i] = opt->best_solution.position[i];
    double final_cost = objective_function(best_pos);
    if (fabs(final_cost - ctx.best_cost) > 1e-5) {
        fprintf(stderr, "Warning: Best solution cost mismatch. Reported: %f, Actual: %f\n", ctx.best_cost, final_cost);
        ctx.best_cost = (float)final_cost;
        opt->best_solution.fitness = final_cost;
    }
    free(best_pos);

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Particles: %.3f\n", prof.init_time);
    printf("Calculate Forces: %.3f\n", prof.forces_time);
    printf("Update Positions: %.3f\n", prof.update_pos_time);
    printf("Update Charged Memory: %.3f\n", prof.update_cm_time);
    printf("Find Min/Max: %.3f\n", prof.min_max_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("\nAverage per Iteration (ms):\n");
        printf("Calculate Forces: %.3f\n", prof.forces_time / prof.count);
        printf("Update Positions: %.3f\n", prof.update_pos_time / prof.count);
        printf("Update Charged Memory: %.3f\n", prof.update_cm_time / prof.count);
        printf("Find Min/Max: %.3f\n", prof.min_max_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Cleanup
    free(positions);
    free(costs);
    CSS_cleanup_context(&ctx, &cl_ctx);
    CSS_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
