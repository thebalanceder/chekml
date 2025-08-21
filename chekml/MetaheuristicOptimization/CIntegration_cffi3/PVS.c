#include "PVS.h"
#include <string.h>
#include <time.h>
#include <float.h>

// Fast Xorshift random number generator state
static uint32_t xorshift_state = 1;

// Initialize Xorshift seed
void init_xorshift_pvs(uint32_t seed) {
    xorshift_state = seed ? seed : (uint32_t)time(NULL);
}

// Fast Xorshift random number generator
static inline uint32_t xorshift32_pvs() {
    uint32_t x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

// Generate random double between min and max
static inline double rand_double_pvs(double min, double max) {
    double r = (double)xorshift32_pvs() / UINT32_MAX;
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

void PVS_init_cl(PVSCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->init_vortex_kernel = clCreateKernel(cl_ctx->program, "init_vortex", &err);
    check_cl_error(err, "clCreateKernel init_vortex");
    cl_ctx->first_phase_kernel = clCreateKernel(cl_ctx->program, "first_phase", &err);
    check_cl_error(err, "clCreateKernel first_phase");
    cl_ctx->crossover_mutation_kernel = clCreateKernel(cl_ctx->program, "crossover_mutation", &err);
    check_cl_error(err, "clCreateKernel crossover_mutation");
}

void PVS_cleanup_cl(PVSCLContext *cl_ctx) {
    if (cl_ctx->crossover_mutation_kernel) clReleaseKernel(cl_ctx->crossover_mutation_kernel);
    if (cl_ctx->first_phase_kernel) clReleaseKernel(cl_ctx->first_phase_kernel);
    if (cl_ctx->init_vortex_kernel) clReleaseKernel(cl_ctx->init_vortex_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void PVS_init_context(PVSContext *ctx, Optimizer *opt, PVSCLContext *cl_ctx) {
    cl_int err;
    ctx->best_cost = INFINITY;

    int population_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->positions = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer positions");
    ctx->costs = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer costs");
    ctx->center = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer center");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->temp_solution = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer temp_solution");
    ctx->mutated_solution = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer mutated_solution");
    ctx->probabilities = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer probabilities");

    // Initialize random seeds
    cl_uint *seeds = (cl_uint *)malloc(population_size * sizeof(cl_uint));
    init_xorshift_pvs((uint32_t)time(NULL));
    for (int i = 0; i < population_size; i++) seeds[i] = xorshift32_pvs();
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->random_seeds, CL_TRUE, 0, population_size * sizeof(cl_uint), seeds, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer seeds");
    free(seeds);

    // Initialize bounds and center
    float *bounds_float = (float *)malloc(dim * 2 * sizeof(float));
    float *center_float = (float *)malloc(dim * sizeof(float));
    for (int j = 0; j < dim; j++) {
        bounds_float[2 * j] = (float)opt->bounds[2 * j]; // Lower bound
        bounds_float[2 * j + 1] = (float)opt->bounds[2 * j + 1]; // Upper bound
        center_float[j] = (float)(0.5 * (opt->bounds[2 * j] + opt->bounds[2 * j + 1]));
    }
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->bounds, CL_TRUE, 0, dim * 2 * sizeof(cl_float), bounds_float, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer bounds");
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->center, CL_TRUE, 0, dim * sizeof(cl_float), center_float, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer center");
    free(bounds_float);
    free(center_float);
}

void PVS_cleanup_context(PVSContext *ctx, PVSCLContext *cl_ctx) {
    if (ctx->probabilities) clReleaseMemObject(ctx->probabilities);
    if (ctx->mutated_solution) clReleaseMemObject(ctx->mutated_solution);
    if (ctx->temp_solution) clReleaseMemObject(ctx->temp_solution);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->center) clReleaseMemObject(ctx->center);
    if (ctx->costs) clReleaseMemObject(ctx->costs);
    if (ctx->positions) clReleaseMemObject(ctx->positions);
}

void PVS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU PVS optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Validate bounds
    int dim = opt->dim;
    for (int j = 0; j < dim; j++) {
        if (opt->bounds[2 * j] >= opt->bounds[2 * j + 1]) {
            fprintf(stderr, "Invalid bounds for dimension %d: lower=%f, upper=%f\n", j, opt->bounds[2 * j], opt->bounds[2 * j + 1]);
            exit(1);
        }
    }

    // Read kernel source
    FILE *fp = fopen("PVS.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open PVS.cl\n");
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
    PVSCLContext cl_ctx = {0};
    PVS_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    PVSContext ctx = {0};
    PVS_init_context(&ctx, opt, &cl_ctx);

    int population_size = opt->population_size;
    int func_count = 0;
    int iteration = 0;
    int stagnation_count = 0;
    float prev_best_cost = INFINITY;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * population_size : PVS_MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_time;
        double first_phase_time;
        double crossover_mutation_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;

    // Initialize vortex
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_vortex_kernel, 0, sizeof(cl_mem), &ctx.positions);
    clSetKernelArg(cl_ctx.init_vortex_kernel, 1, sizeof(cl_mem), &ctx.center);
    clSetKernelArg(cl_ctx.init_vortex_kernel, 2, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_vortex_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_vortex_kernel, 4, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_vortex_kernel, 5, sizeof(cl_int), &population_size);
    size_t global_work_size = population_size;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.init_vortex_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
    check_cl_error(err, "clEnqueueNDRangeKernel init_vortex");
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
        for (int j = 0; j < dim; j++) {
            double orig_pos = (double)positions[i * dim + j];
            pos[j] = fmax(opt->bounds[2 * j] - 1e-10, fmin(opt->bounds[2 * j + 1] + 1e-10, orig_pos));
            if (fabs(orig_pos - pos[j]) > 1e-10) {
                fprintf(stderr, "Bound violation at particle %d, dim %d: %f adjusted to %f\n", i, j, orig_pos, pos[j]);
            }
        }
        costs[i] = (float)objective_function(pos);
        costs[i] = isnan(costs[i]) || isinf(costs[i]) ? INFINITY : costs[i];
        if (costs[i] < ctx.best_cost) {
            ctx.best_cost = costs[i];
            opt->best_solution.fitness = ctx.best_cost;
            for (int j = 0; j < dim; j++) {
                opt->best_solution.position[j] = pos[j];
            }
            float *center_update = (float *)malloc(dim * sizeof(float));
            for (int j = 0; j < dim; j++) {
                center_update[j] = (float)pos[j];
            }
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.center, CL_TRUE, 0, dim * sizeof(cl_float), center_update, 0, NULL, NULL);
            check_cl_error(err, "clEnqueueWriteBuffer center");
            free(center_update);
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

    // Main optimization loop
    while (func_count < max_evals) {
        iteration++;

        // Update radius
        float a = (opt->max_iter - iteration) / (float)opt->max_iter;
        a = fmax(a, 0.1f);
        float ginv = (1.0f / PVS_X_GAMMA) * (1.0f / (PVS_X_GAMMA * (0.5f + 0.5f * a)));
        float radius = ginv * (float)(opt->bounds[1] - opt->bounds[0]) / 2.0f;

        // First phase
        cl_event first_phase_event;
        int size = (iteration == 0) ? population_size : population_size / 2;
        clSetKernelArg(cl_ctx.first_phase_kernel, 0, sizeof(cl_mem), &ctx.positions);
        clSetKernelArg(cl_ctx.first_phase_kernel, 1, sizeof(cl_mem), &ctx.center);
        clSetKernelArg(cl_ctx.first_phase_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.first_phase_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.first_phase_kernel, 4, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.first_phase_kernel, 5, sizeof(cl_int), &size);
        clSetKernelArg(cl_ctx.first_phase_kernel, 6, sizeof(cl_float), &radius);
        global_work_size = size;
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.first_phase_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &first_phase_event);
        check_cl_error(err, "clEnqueueNDRangeKernel first_phase");
        clFinish(cl_ctx.queue);

        if (first_phase_event) {
            clGetEventProfilingInfo(first_phase_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(first_phase_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            prof.first_phase_time += (end - start) / 1e6;
            clReleaseEvent(first_phase_event);
        }

        // Evaluate first phase population
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer positions");
        for (int i = 0; i < size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) {
                double orig_pos = (double)positions[i * dim + j];
                pos[j] = fmax(opt->bounds[2 * j] - 1e-10, fmin(opt->bounds[2 * j + 1] + 1e-10, orig_pos));
                if (fabs(orig_pos - pos[j]) > 1e-10) {
                    fprintf(stderr, "Bound violation at particle %d, dim %d: %f adjusted to %f\n", i, j, orig_pos, pos[j]);
                }
            }
            costs[i] = (float)objective_function(pos);
            costs[i] = isnan(costs[i]) || isinf(costs[i]) ? INFINITY : costs[i];
            if (costs[i] < ctx.best_cost) {
                ctx.best_cost = costs[i];
                opt->best_solution.fitness = ctx.best_cost;
                for (int j = 0; j < dim; j++) {
                    opt->best_solution.position[j] = pos[j];
                }
                float *center_update = (float *)malloc(dim * sizeof(float));
                for (int j = 0; j < dim; j++) {
                    center_update[j] = (float)pos[j];
                }
                err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.center, CL_TRUE, 0, dim * sizeof(cl_float), center_update, 0, NULL, NULL);
                check_cl_error(err, "clEnqueueWriteBuffer center");
                free(center_update);
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

        func_count += size;

        // Second phase: Crossover and Mutation
        cl_event crossover_mutation_event;
        float prob_mut = 1.0f / dim;
        float prob_cross = 1.0f / dim;
        clSetKernelArg(cl_ctx.crossover_mutation_kernel, 0, sizeof(cl_mem), &ctx.positions);
        clSetKernelArg(cl_ctx.crossover_mutation_kernel, 1, sizeof(cl_mem), &ctx.costs);
        clSetKernelArg(cl_ctx.crossover_mutation_kernel, 2, sizeof(cl_mem), &ctx.probabilities);
        clSetKernelArg(cl_ctx.crossover_mutation_kernel, 3, sizeof(cl_mem), &ctx.temp_solution);
        clSetKernelArg(cl_ctx.crossover_mutation_kernel, 4, sizeof(cl_mem), &ctx.mutated_solution);
        clSetKernelArg(cl_ctx.crossover_mutation_kernel, 5, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.crossover_mutation_kernel, 6, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.crossover_mutation_kernel, 7, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.crossover_mutation_kernel, 8, sizeof(cl_int), &population_size);
        clSetKernelArg(cl_ctx.crossover_mutation_kernel, 9, sizeof(cl_float), &prob_mut);
        clSetKernelArg(cl_ctx.crossover_mutation_kernel, 10, sizeof(cl_float), &prob_cross);
        global_work_size = population_size / 2;
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.crossover_mutation_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &crossover_mutation_event);
        check_cl_error(err, "clEnqueueNDRangeKernel crossover_mutation");
        clFinish(cl_ctx.queue);

        if (crossover_mutation_event) {
            clGetEventProfilingInfo(crossover_mutation_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(crossover_mutation_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            prof.crossover_mutation_time += (end - start) / 1e6;
            clReleaseEvent(crossover_mutation_event);
        }

        // Evaluate second phase population
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer positions");
        for (int i = population_size / 2; i < population_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) {
                double orig_pos = (double)positions[i * dim + j];
                pos[j] = fmax(opt->bounds[2 * j] - 1e-10, fmin(opt->bounds[2 * j + 1] + 1e-10, orig_pos));
                if (fabs(orig_pos - pos[j]) > 1e-10) {
                    fprintf(stderr, "Bound violation at particle %d, dim %d: %f adjusted to %f\n", i, j, orig_pos, pos[j]);
                }
            }
            costs[i] = (float)objective_function(pos);
            costs[i] = isnan(costs[i]) || isinf(costs[i]) ? INFINITY : costs[i];
            if (costs[i] < ctx.best_cost) {
                ctx.best_cost = costs[i];
                opt->best_solution.fitness = ctx.best_cost;
                for (int j = 0; j < dim; j++) {
                    opt->best_solution.position[j] = pos[j];
                }
                float *center_update = (float *)malloc(dim * sizeof(float));
                for (int j = 0; j < dim; j++) {
                    center_update[j] = (float)pos[j];
                }
                err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.center, CL_TRUE, 0, dim * sizeof(cl_float), center_update, 0, NULL, NULL);
                check_cl_error(err, "clEnqueueWriteBuffer center");
                free(center_update);
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

        func_count += population_size / 2;

        // Convergence check
        if (fabs(prev_best_cost - ctx.best_cost) < PVS_CONVERGENCE_TOL) {
            stagnation_count++;
            if (stagnation_count >= PVS_STAGNATION_THRESHOLD) {
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
    for (int i = 0; i < dim; i++) {
        best_pos[i] = opt->best_solution.position[i];
        best_pos[i] = fmax(opt->bounds[2 * i] - 1e-10, fmin(opt->bounds[2 * i + 1] + 1e-10, best_pos[i]));
        if (fabs(best_pos[i] - opt->best_solution.position[i]) > 1e-10) {
            fprintf(stderr, "Bound violation in final best solution, dim %d: %f adjusted to %f\n", i, opt->best_solution.position[i], best_pos[i]);
        }
    }
    double final_cost = objective_function(best_pos);
    if (fabs(final_cost - ctx.best_cost) > 1e-5) {
        fprintf(stderr, "Warning: Best solution cost mismatch. Reported: %f, Actual: %f\n", ctx.best_cost, final_cost);
        ctx.best_cost = (float)final_cost;
        opt->best_solution.fitness = final_cost;
        for (int i = 0; i < dim; i++) {
            opt->best_solution.position[i] = best_pos[i];
        }
    }
    free(best_pos);

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Vortex: %.3f\n", prof.init_time);
    printf("First Phase: %.3f\n", prof.first_phase_time);
    printf("Crossover and Mutation: %.3f\n", prof.crossover_mutation_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("\nAverage per Iteration (ms):\n");
        printf("First Phase: %.3f\n", prof.first_phase_time / prof.count);
        printf("Crossover and Mutation: %.3f\n", prof.crossover_mutation_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Cleanup
    free(positions);
    free(costs);
    PVS_cleanup_context(&ctx, &cl_ctx);
    PVS_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
