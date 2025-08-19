#include "TFWO.h"
#include <string.h>
#include <time.h>
#include <float.h>

// Fast Xorshift random number generator state
static uint32_t xorshift_state = 1;

// Initialize Xorshift seed
void init_xorshift_tfwo(uint32_t seed) {
    xorshift_state = seed ? seed : (uint32_t)time(NULL);
}

// Fast Xorshift random number generator
static inline uint32_t xorshift32_tfwo() {
    uint32_t x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

// Generate random double between min and max
static inline double rand_double_tfwo(double min, double max) {
    double r = (double)xorshift32_tfwo() / UINT32_MAX;
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

void TFWO_init_cl(TFWOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->init_whirlpools_kernel = clCreateKernel(cl_ctx->program, "init_whirlpools", &err);
    check_cl_error(err, "clCreateKernel init_whirlpools");
    cl_ctx->effects_whirlpools_kernel = clCreateKernel(cl_ctx->program, "effects_whirlpools", &err);
    check_cl_error(err, "clCreateKernel effects_whirlpools");
    cl_ctx->update_best_kernel = clCreateKernel(cl_ctx->program, "update_best", &err);
    check_cl_error(err, "clCreateKernel update_best");
}

void TFWO_cleanup_cl(TFWOCLContext *cl_ctx) {
    if (cl_ctx->update_best_kernel) clReleaseKernel(cl_ctx->update_best_kernel);
    if (cl_ctx->effects_whirlpools_kernel) clReleaseKernel(cl_ctx->effects_whirlpools_kernel);
    if (cl_ctx->init_whirlpools_kernel) clReleaseKernel(cl_ctx->init_whirlpools_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void TFWO_init_context(TFWOContext *ctx, Optimizer *opt, TFWOCLContext *cl_ctx) {
    cl_int err;
    ctx->best_cost = INFINITY;

    int n_whirlpools = N_WHIRLPOOLS_DEFAULT;
    int n_objects_per_whirlpool = N_OBJECTS_PER_WHIRLPOOL_DEFAULT;
    int total_objects = n_whirlpools * n_objects_per_whirlpool;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->wp_positions = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, n_whirlpools * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer wp_positions");
    ctx->wp_costs = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, n_whirlpools * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer wp_costs");
    ctx->wp_deltas = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, n_whirlpools * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer wp_deltas");
    ctx->wp_position_sums = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, n_whirlpools * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer wp_position_sums");
    ctx->obj_positions = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, total_objects * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer obj_positions");
    ctx->obj_costs = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, total_objects * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer obj_costs");
    ctx->obj_deltas = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, total_objects * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer obj_deltas");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, (n_whirlpools + total_objects) * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->temp_d = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, total_objects * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer temp_d");
    ctx->temp_d2 = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, total_objects * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer temp_d2");
    ctx->temp_RR = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, total_objects * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer temp_RR");
    ctx->temp_J = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, n_whirlpools * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer temp_J");

    // Initialize random seeds
    cl_uint *seeds = (cl_uint *)malloc((n_whirlpools + total_objects) * sizeof(cl_uint));
    init_xorshift_tfwo((uint32_t)time(NULL));
    for (int i = 0; i < n_whirlpools + total_objects; i++) seeds[i] = xorshift32_tfwo();
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->random_seeds, CL_TRUE, 0, (n_whirlpools + total_objects) * sizeof(cl_uint), seeds, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer seeds");
    free(seeds);

    // Initialize bounds
    float *bounds_float = (float *)malloc(dim * 2 * sizeof(float));
    for (int i = 0; i < dim * 2; i++) bounds_float[i] = (float)opt->bounds[i];
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->bounds, CL_TRUE, 0, dim * 2 * sizeof(cl_float), bounds_float, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer bounds");
    free(bounds_float);

    // Initialize deltas to zero
    float *zero_deltas = (float *)calloc(n_whirlpools + total_objects, sizeof(float));
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->wp_deltas, CL_TRUE, 0, n_whirlpools * sizeof(cl_float), zero_deltas, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer wp_deltas");
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->obj_deltas, CL_TRUE, 0, total_objects * sizeof(cl_float), zero_deltas, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer obj_deltas");
    free(zero_deltas);
}

void TFWO_cleanup_context(TFWOContext *ctx, TFWOCLContext *cl_ctx) {
    if (ctx->temp_J) clReleaseMemObject(ctx->temp_J);
    if (ctx->temp_RR) clReleaseMemObject(ctx->temp_RR);
    if (ctx->temp_d2) clReleaseMemObject(ctx->temp_d2);
    if (ctx->temp_d) clReleaseMemObject(ctx->temp_d);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->obj_deltas) clReleaseMemObject(ctx->obj_deltas);
    if (ctx->obj_costs) clReleaseMemObject(ctx->obj_costs);
    if (ctx->obj_positions) clReleaseMemObject(ctx->obj_positions);
    if (ctx->wp_position_sums) clReleaseMemObject(ctx->wp_position_sums);
    if (ctx->wp_deltas) clReleaseMemObject(ctx->wp_deltas);
    if (ctx->wp_costs) clReleaseMemObject(ctx->wp_costs);
    if (ctx->wp_positions) clReleaseMemObject(ctx->wp_positions);
}

void TFWO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU TFWO optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("TFWO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open TFWO.cl\n");
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
    TFWOCLContext cl_ctx = {0};
    TFWO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    TFWOContext ctx = {0};
    TFWO_init_context(&ctx, opt, &cl_ctx);

    int n_whirlpools = N_WHIRLPOOLS_DEFAULT;
    int n_objects_per_whirlpool = N_OBJECTS_PER_WHIRLPOOL_DEFAULT;
    int total_objects = n_whirlpools * n_objects_per_whirlpool;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;
    int stagnation_count = 0;
    float prev_best_cost = INFINITY;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * (n_whirlpools + total_objects) : TFWO_MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_time;
        double effects_time;
        double update_best_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;

    // Initialize whirlpools and objects
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_whirlpools_kernel, 0, sizeof(cl_mem), &ctx.wp_positions);
    clSetKernelArg(cl_ctx.init_whirlpools_kernel, 1, sizeof(cl_mem), &ctx.wp_costs);
    clSetKernelArg(cl_ctx.init_whirlpools_kernel, 2, sizeof(cl_mem), &ctx.wp_deltas);
    clSetKernelArg(cl_ctx.init_whirlpools_kernel, 3, sizeof(cl_mem), &ctx.wp_position_sums);
    clSetKernelArg(cl_ctx.init_whirlpools_kernel, 4, sizeof(cl_mem), &ctx.obj_positions);
    clSetKernelArg(cl_ctx.init_whirlpools_kernel, 5, sizeof(cl_mem), &ctx.obj_costs);
    clSetKernelArg(cl_ctx.init_whirlpools_kernel, 6, sizeof(cl_mem), &ctx.obj_deltas);
    clSetKernelArg(cl_ctx.init_whirlpools_kernel, 7, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_whirlpools_kernel, 8, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_whirlpools_kernel, 9, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_whirlpools_kernel, 10, sizeof(cl_int), &n_whirlpools);
    clSetKernelArg(cl_ctx.init_whirlpools_kernel, 11, sizeof(cl_int), &n_objects_per_whirlpool);
    size_t global_work_size = n_whirlpools + total_objects;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.init_whirlpools_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
    check_cl_error(err, "clEnqueueNDRangeKernel init_whirlpools");
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
    float *wp_positions = (float *)malloc(n_whirlpools * dim * sizeof(float));
    float *wp_costs = (float *)malloc(n_whirlpools * sizeof(float));
    float *obj_positions = (float *)malloc(total_objects * dim * sizeof(float));
    float *obj_costs = (float *)malloc(total_objects * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.wp_positions, CL_TRUE, 0, n_whirlpools * dim * sizeof(cl_float), wp_positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer wp_positions");
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.obj_positions, CL_TRUE, 0, total_objects * dim * sizeof(cl_float), obj_positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueReadBuffer obj_positions");

    for (int i = 0; i < n_whirlpools; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = (double)wp_positions[i * dim + j];
        wp_costs[i] = (float)objective_function(pos);
        wp_costs[i] = isnan(wp_costs[i]) || isinf(wp_costs[i]) ? INFINITY : wp_costs[i];
        if (wp_costs[i] < ctx.best_cost) {
            ctx.best_cost = wp_costs[i];
            opt->best_solution.fitness = ctx.best_cost;
            for (int j = 0; j < dim; j++) opt->best_solution.position[j] = pos[j];
        }
        free(pos);
    }
    for (int i = 0; i < total_objects; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = (double)obj_positions[i * dim + j];
        obj_costs[i] = (float)objective_function(pos);
        obj_costs[i] = isnan(obj_costs[i]) || isinf(obj_costs[i]) ? INFINITY : obj_costs[i];
        if (obj_costs[i] < ctx.best_cost) {
            ctx.best_cost = obj_costs[i];
            opt->best_solution.fitness = ctx.best_cost;
            for (int j = 0; j < dim; j++) opt->best_solution.position[j] = pos[j];
        }
        free(pos);
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.wp_costs, CL_TRUE, 0, n_whirlpools * sizeof(cl_float), wp_costs, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer wp_costs");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.obj_costs, CL_TRUE, 0, total_objects * sizeof(cl_float), obj_costs, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer obj_costs");

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

    func_count += n_whirlpools + total_objects;

    // Main optimization loop
    while (func_count < max_evals) {
        iteration++;

        // Compute whirlpool effects
        cl_event effects_event;
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 0, sizeof(cl_mem), &ctx.wp_positions);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 1, sizeof(cl_mem), &ctx.wp_costs);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 2, sizeof(cl_mem), &ctx.wp_deltas);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 3, sizeof(cl_mem), &ctx.wp_position_sums);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 4, sizeof(cl_mem), &ctx.obj_positions);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 5, sizeof(cl_mem), &ctx.obj_costs);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 6, sizeof(cl_mem), &ctx.obj_deltas);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 7, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 8, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 9, sizeof(cl_mem), &ctx.temp_d);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 10, sizeof(cl_mem), &ctx.temp_d2);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 11, sizeof(cl_mem), &ctx.temp_RR);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 12, sizeof(cl_mem), &ctx.temp_J);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 13, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 14, sizeof(cl_int), &n_whirlpools);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 15, sizeof(cl_int), &n_objects_per_whirlpool);
        clSetKernelArg(cl_ctx.effects_whirlpools_kernel, 16, sizeof(cl_int), &iteration);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.effects_whirlpools_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &effects_event);
        check_cl_error(err, "clEnqueueNDRangeKernel effects_whirlpools");
        clFinish(cl_ctx.queue);

        if (effects_event) {
            clGetEventProfilingInfo(effects_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(effects_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            prof.effects_time += (end - start) / 1e6;
            clReleaseEvent(effects_event);
        }

        // Update best whirlpools
        cl_event update_best_event;
        clSetKernelArg(cl_ctx.update_best_kernel, 0, sizeof(cl_mem), &ctx.wp_positions);
        clSetKernelArg(cl_ctx.update_best_kernel, 1, sizeof(cl_mem), &ctx.wp_costs);
        clSetKernelArg(cl_ctx.update_best_kernel, 2, sizeof(cl_mem), &ctx.wp_position_sums);
        clSetKernelArg(cl_ctx.update_best_kernel, 3, sizeof(cl_mem), &ctx.obj_positions);
        clSetKernelArg(cl_ctx.update_best_kernel, 4, sizeof(cl_mem), &ctx.obj_costs);
        clSetKernelArg(cl_ctx.update_best_kernel, 5, sizeof(cl_mem), &ctx.temp_RR);
        clSetKernelArg(cl_ctx.update_best_kernel, 6, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.update_best_kernel, 7, sizeof(cl_int), &n_whirlpools);
        clSetKernelArg(cl_ctx.update_best_kernel, 8, sizeof(cl_int), &n_objects_per_whirlpool);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_best_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &update_best_event);
        check_cl_error(err, "clEnqueueNDRangeKernel update_best");
        clFinish(cl_ctx.queue);

        if (update_best_event) {
            clGetEventProfilingInfo(update_best_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(update_best_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            prof.update_best_time += (end - start) / 1e6;
            clReleaseEvent(update_best_event);
        }

        // Evaluate updated population
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.wp_positions, CL_TRUE, 0, n_whirlpools * dim * sizeof(cl_float), wp_positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer wp_positions");
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.obj_positions, CL_TRUE, 0, total_objects * dim * sizeof(cl_float), obj_positions, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer obj_positions");

        for (int i = 0; i < n_whirlpools; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = (double)wp_positions[i * dim + j];
            wp_costs[i] = (float)objective_function(pos);
            wp_costs[i] = isnan(wp_costs[i]) || isinf(wp_costs[i]) ? INFINITY : wp_costs[i];
            if (wp_costs[i] < ctx.best_cost) {
                ctx.best_cost = wp_costs[i];
                opt->best_solution.fitness = ctx.best_cost;
                for (int j = 0; j < dim; j++) opt->best_solution.position[j] = pos[j];
            }
            free(pos);
        }
        for (int i = 0; i < total_objects; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = (double)obj_positions[i * dim + j];
            obj_costs[i] = (float)objective_function(pos);
            obj_costs[i] = isnan(obj_costs[i]) || isinf(obj_costs[i]) ? INFINITY : obj_costs[i];
            if (obj_costs[i] < ctx.best_cost) {
                ctx.best_cost = obj_costs[i];
                opt->best_solution.fitness = ctx.best_cost;
                for (int j = 0; j < dim; j++) opt->best_solution.position[j] = pos[j];
            }
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.wp_costs, CL_TRUE, 0, n_whirlpools * sizeof(cl_float), wp_costs, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer wp_costs");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.obj_costs, CL_TRUE, 0, total_objects * sizeof(cl_float), obj_costs, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer obj_costs");
        func_count += n_whirlpools + total_objects;

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

        // Convergence check
        if (fabs(prev_best_cost - ctx.best_cost) < TFWO_CONVERGENCE_TOL) {
            stagnation_count++;
            if (stagnation_count >= TFWO_STAGNATION_THRESHOLD) {
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
    printf("Initialize Whirlpools: %.3f\n", prof.init_time);
    printf("Compute Whirlpool Effects: %.3f\n", prof.effects_time);
    printf("Update Best Whirlpools: %.3f\n", prof.update_best_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("\nAverage per Iteration (ms):\n");
        printf("Compute Whirlpool Effects: %.3f\n", prof.effects_time / prof.count);
        printf("Update Best Whirlpools: %.3f\n", prof.update_best_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Cleanup
    free(wp_positions);
    free(wp_costs);
    free(obj_positions);
    free(obj_costs);
    TFWO_cleanup_context(&ctx, &cl_ctx);
    TFWO_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
