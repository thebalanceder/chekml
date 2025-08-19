#include "BHA.h"
#include <string.h>
#include <time.h>
#include <float.h>

// Fast Xorshift random number generator state
static uint32_t xorshift_state = 1;

// Initialize Xorshift seed
void init_xorshift_bha(uint32_t seed) {
    xorshift_state = seed ? seed : (uint32_t)time(NULL);
}

// Fast Xorshift random number generator
static inline uint32_t xorshift32_bha() {
    uint32_t x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

// Generate random double between min and max
static inline double rand_double_bha(double min, double max) {
    double r = (double)xorshift32_bha() / UINT32_MAX;
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

void BHA_init_cl(BHACLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->init_stars_kernel = clCreateKernel(cl_ctx->program, "init_stars", &err);
    check_cl_error(err, "clCreateKernel init_stars");
    cl_ctx->update_positions_kernel = clCreateKernel(cl_ctx->program, "update_positions", &err);
    check_cl_error(err, "clCreateKernel update_positions");
    cl_ctx->new_star_gen_kernel = clCreateKernel(cl_ctx->program, "new_star_gen", &err);
    check_cl_error(err, "clCreateKernel new_star_gen");
    cl_ctx->find_black_hole_kernel = clCreateKernel(cl_ctx->program, "find_black_hole", &err);
    check_cl_error(err, "clCreateKernel find_black_hole");
}

void BHA_cleanup_cl(BHACLContext *cl_ctx) {
    if (cl_ctx->find_black_hole_kernel) clReleaseKernel(cl_ctx->find_black_hole_kernel);
    if (cl_ctx->new_star_gen_kernel) clReleaseKernel(cl_ctx->new_star_gen_kernel);
    if (cl_ctx->update_positions_kernel) clReleaseKernel(cl_ctx->update_positions_kernel);
    if (cl_ctx->init_stars_kernel) clReleaseKernel(cl_ctx->init_stars_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void BHA_init_context(BHAContext *ctx, Optimizer *opt, BHACLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;

    int population_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->positions = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer positions");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->best_index = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_index");

    // Initialize random seeds
    cl_uint *seeds = (cl_uint *)malloc(population_size * sizeof(cl_uint));
    init_xorshift_bha((uint32_t)time(NULL));
    for (int i = 0; i < population_size; i++) seeds[i] = xorshift32_bha();
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->random_seeds, CL_TRUE, 0, population_size * sizeof(cl_uint), seeds, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer seeds");
    free(seeds);

    // Initialize bounds
    float *bounds_float = (float *)malloc(dim * 2 * sizeof(float));
    for (int i = 0; i < dim * 2; i++) bounds_float[i] = (float)opt->bounds[i];
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->bounds, CL_TRUE, 0, dim * 2 * sizeof(cl_float), bounds_float, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer bounds");
    free(bounds_float);
}

void BHA_cleanup_context(BHAContext *ctx, BHACLContext *cl_ctx) {
    if (ctx->best_index) clReleaseMemObject(ctx->best_index);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->positions) clReleaseMemObject(ctx->positions);
}

void BHA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU BHA optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("BHA.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open BHA.cl\n");
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
    BHACLContext cl_ctx = {0};
    BHA_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    BHAContext ctx = {0};
    BHA_init_context(&ctx, opt, &cl_ctx);

    int population_size = opt->population_size;
    int dim = opt->dim;
    int max_iter = opt->max_iter > 0 ? opt->max_iter : BHA_MAX_ITER;
    cl_int black_hole_idx = 0;

    // Profiling data
    typedef struct {
        double init_time;
        double update_pos_time;
        double new_star_time;
        double find_bh_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;

    // Initialize stars
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_stars_kernel, 0, sizeof(cl_mem), &ctx.positions);
    clSetKernelArg(cl_ctx.init_stars_kernel, 1, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_stars_kernel, 2, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_stars_kernel, 3, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_stars_kernel, 4, sizeof(cl_int), &population_size);
    size_t global_work_size = population_size;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.init_stars_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
    check_cl_error(err, "clEnqueueNDRangeKernel init_stars");
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
    float *fitness = (float *)malloc(population_size * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer positions");
    for (int i = 0; i < population_size; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = (double)positions[i * dim + j];
        fitness[i] = (float)objective_function(pos);
        fitness[i] = isnan(fitness[i]) || isinf(fitness[i]) ? INFINITY : fitness[i];
        if (fitness[i] < ctx.best_fitness) {
            ctx.best_fitness = fitness[i];
            black_hole_idx = i;
            opt->best_solution.fitness = ctx.best_fitness;
            for (int j = 0; j < dim; j++) opt->best_solution.position[j] = pos[j];
        }
        free(pos);
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, population_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer fitness");

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

    // Write initial black hole index
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_index, CL_TRUE, 0, sizeof(cl_int), &black_hole_idx, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer best_index");

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Update positions
        cl_event update_pos_event;
        clSetKernelArg(cl_ctx.update_positions_kernel, 0, sizeof(cl_mem), &ctx.positions);
        clSetKernelArg(cl_ctx.update_positions_kernel, 1, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.update_positions_kernel, 2, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.update_positions_kernel, 3, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.update_positions_kernel, 4, sizeof(cl_int), &population_size);
        clSetKernelArg(cl_ctx.update_positions_kernel, 5, sizeof(cl_int), &black_hole_idx);
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
            fitness[i] = (float)objective_function(pos);
            fitness[i] = isnan(fitness[i]) || isinf(fitness[i]) ? INFINITY : fitness[i];
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, population_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer fitness");

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

        // Find new black hole
        cl_event find_bh_event;
        clSetKernelArg(cl_ctx.find_black_hole_kernel, 0, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.find_black_hole_kernel, 1, sizeof(cl_mem), &ctx.best_index);
        clSetKernelArg(cl_ctx.find_black_hole_kernel, 2, sizeof(cl_int), &population_size);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.find_black_hole_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &find_bh_event);
        check_cl_error(err, "clEnqueueNDRangeKernel find_black_hole");
        clFinish(cl_ctx.queue);

        if (find_bh_event) {
            clGetEventProfilingInfo(find_bh_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(find_bh_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            prof.find_bh_time += (end - start) / 1e6;
            clReleaseEvent(find_bh_event);
        }

        // Read new black hole index and update best solution
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_index, CL_TRUE, 0, sizeof(cl_int), &black_hole_idx, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer best_index");
        if (fitness[black_hole_idx] < ctx.best_fitness) {
            ctx.best_fitness = fitness[black_hole_idx];
            opt->best_solution.fitness = ctx.best_fitness;
            for (int j = 0; j < dim; j++) {
                opt->best_solution.position[j] = (double)positions[black_hole_idx * dim + j];
            }
        }

        if (read_event) {
            clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            prof.read_time += (end - start) / 1e6;
            clReleaseEvent(read_event);
        }

        // Generate new stars
        cl_event new_star_event;
        clSetKernelArg(cl_ctx.new_star_gen_kernel, 0, sizeof(cl_mem), &ctx.positions);
        clSetKernelArg(cl_ctx.new_star_gen_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.new_star_gen_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.new_star_gen_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.new_star_gen_kernel, 4, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.new_star_gen_kernel, 5, sizeof(cl_int), &population_size);
        clSetKernelArg(cl_ctx.new_star_gen_kernel, 6, sizeof(cl_int), &black_hole_idx);
        clSetKernelArg(cl_ctx.new_star_gen_kernel, 7, sizeof(cl_float), &ctx.best_fitness);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.new_star_gen_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &new_star_event);
        check_cl_error(err, "clEnqueueNDRangeKernel new_star_gen");
        clFinish(cl_ctx.queue);

        if (new_star_event) {
            clGetEventProfilingInfo(new_star_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(new_star_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            prof.new_star_time += (end - start) / 1e6;
            clReleaseEvent(new_star_event);
        }

        // Evaluate new stars
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer positions");
        for (int i = 0; i < population_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = (double)positions[i * dim + j];
            fitness[i] = (float)objective_function(pos);
            fitness[i] = isnan(fitness[i]) || isinf(fitness[i]) ? INFINITY : fitness[i];
            if (fitness[i] < ctx.best_fitness) {
                ctx.best_fitness = fitness[i];
                black_hole_idx = i;
                opt->best_solution.fitness = ctx.best_fitness;
                for (int j = 0; j < dim; j++) opt->best_solution.position[j] = pos[j];
            }
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, population_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer fitness");

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

        prof.count++;
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, ctx.best_fitness);
    }

    // Validate final best solution
    double *best_pos = (double *)malloc(dim * sizeof(double));
    for (int i = 0; i < dim; i++) best_pos[i] = opt->best_solution.position[i];
    double final_cost = objective_function(best_pos);
    if (fabs(final_cost - ctx.best_fitness) > 1e-5) {
        fprintf(stderr, "Warning: Best solution cost mismatch. Reported: %f, Actual: %f\n", ctx.best_fitness, final_cost);
        ctx.best_fitness = (float)final_cost;
        opt->best_solution.fitness = final_cost;
    }
    free(best_pos);

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Stars: %.3f\n", prof.init_time);
    printf("Update Positions: %.3f\n", prof.update_pos_time);
    printf("New Star Generation: %.3f\n", prof.new_star_time);
    printf("Find Black Hole: %.3f\n", prof.find_bh_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("\nAverage per Iteration (ms):\n");
        printf("Update Positions: %.3f\n", prof.update_pos_time / prof.count);
        printf("New Star Generation: %.3f\n", prof.new_star_time / prof.count);
        printf("Find Black Hole: %.3f\n", prof.find_bh_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Cleanup
    free(positions);
    free(fitness);
    BHA_cleanup_context(&ctx, &cl_ctx);
    BHA_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
