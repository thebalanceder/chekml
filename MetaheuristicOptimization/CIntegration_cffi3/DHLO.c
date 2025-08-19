#include "DHLO.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (stub, as no specific preprocessing needed)
char *preprocess_kernel_source(const char *source);

void DHLO_init_cl(DHLCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_command_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_ctx->cl_ctx = clCreateCommandQueueWithProperties(opt->context, opt->device, props, &err);
    check_cl_error(err, "clCreateCommandQueueWithProperties");
    cl_ctx->owns_queue = CL_TRUE;

    // Reuse OpenCL context and device
    cl_ctx->context = opt->context;
    cl_ctx->dev = opt->device;

    // Preprocess and build program
    char *processed_source = preprocess_kernel_source(kernel_source);
    cl_ctx->cl_program = clCreateProgramWithSource(cl_ctx->context, 1, (const char **)&processed_source, NULL, &err);
    check_cl_error(err, "clCreateProgramWithSource");

    const char *build_options = "-cl-std=CL1.2 -cl-mad-enable";
    err = clBuildProgram(cl_ctx->cl_program, 1, &cl_ctx->dev, build_options, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(cl_ctx->cl_program, cl_ctx->dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size + 1);
        if (log) {
            clGetProgramBuildInfo(cl_ctx->cl_program, cl_ctx->dev, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            log[log_size] = '\0';
            fprintf(stderr, "Error building program: %d\nBuild Log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(cl_ctx->cl_program);
        free(processed_source);
        exit(1);
    }
    free(processed_source);

    // Create kernels
    cl_ctx->init_kernel = clCreateKernel(cl_ctx->cl_program, "initialize_population", &err);
    check_cl_error(err, "clCreateKernel initialize_population");
    cl_ctx->update_kernel = clCreateKernel(cl_ctx->cl_program, "update_positions_gwo", &err);
    check_cl_error(err, "clCreateKernel update_positions_gwo");
}

void DHLO_cleanup_cl(DHLCLContext *cl_ctx) {
    if (cl_ctx->update_kernel) clReleaseKernel(cl_ctx->update_kernel);
    if (cl_ctx->init_kernel) clReleaseKernel(cl_ctx->init_kernel);
    if (cl_ctx->cl_program) clReleaseProgram(cl_ctx->cl_program);
    if (cl_ctx->cl_ctx && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->cl_ctx);
}

void DHLO_init_context(DHLOContext *ctx, Optimizer *opt, DHLCLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;

    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->population = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer population");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->leaders = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, INITIAL_LEADERS * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer leaders");
    ctx->leader_fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, INITIAL_LEADERS * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer leader_fitness");
    ctx->pbest = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer pbest");
    ctx->pbest_fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer pbest_fitness");
    ctx->best_solution = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_solution");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");

    // Initialize random seeds
    cl_uint *seeds = (cl_uint *)malloc(pop_size * sizeof(cl_uint));
    srand(time(NULL));
    for (int i = 0; i < pop_size; i++) seeds[i] = rand();
    err = clEnqueueWriteBuffer(cl_ctx->cl_ctx, ctx->random_seeds, CL_TRUE, 0, pop_size * sizeof(cl_uint), seeds, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer random_seeds");
    free(seeds);

    // Initialize bounds
    float *bounds_float = (float *)malloc(dim * 2 * sizeof(float));
    for (int i = 0; i < dim * 2; i++) bounds_float[i] = (float)opt->bounds[i];
    err = clEnqueueWriteBuffer(cl_ctx->cl_ctx, ctx->bounds, CL_TRUE, 0, dim * 2 * sizeof(cl_float), bounds_float, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer bounds");
    free(bounds_float);
}

void DHLO_cleanup_context(DHLOContext *ctx, DHLCLContext *cl_ctx) {
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_solution) clReleaseMemObject(ctx->best_solution);
    if (ctx->pbest_fitness) clReleaseMemObject(ctx->pbest_fitness);
    if (ctx->pbest) clReleaseMemObject(ctx->pbest);
    if (ctx->leader_fitness) clReleaseMemObject(ctx->leader_fitness);
    if (ctx->leaders) clReleaseMemObject(ctx->leaders);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

void DHLO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU DHLO optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("DHLO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open DHLO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
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
        fprintf(stderr, "Error: Failed to read DHLO.cl completely\n");
        free(source_str);
        exit(1);
    }
    source_str[source_size] = '\0';

    // Initialize OpenCL context
    DHLCLContext cl_ctx = {0};
    DHLO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    DHLOContext ctx = {0};
    DHLO_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;
    int num_leaders = INITIAL_LEADERS;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * pop_size : MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_time;
        double update_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;
    cl_mem leader_fitness_history = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, INITIAL_LEADERS * opt->max_iter * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer leader_fitness_history");

    // Initialize population
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_kernel, 0, sizeof(cl_mem), &ctx.population);
    clSetKernelArg(cl_ctx.init_kernel, 1, sizeof(cl_mem), &ctx.fitness);
    clSetKernelArg(cl_ctx.init_kernel, 2, sizeof(cl_mem), &ctx.leaders);
    clSetKernelArg(cl_ctx.init_kernel, 3, sizeof(cl_mem), &ctx.leader_fitness);
    clSetKernelArg(cl_ctx.init_kernel, 4, sizeof(cl_mem), &ctx.pbest);
    clSetKernelArg(cl_ctx.init_kernel, 5, sizeof(cl_mem), &ctx.pbest_fitness);
    clSetKernelArg(cl_ctx.init_kernel, 6, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_kernel, 7, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_kernel, 8, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_kernel, 9, sizeof(cl_int), &pop_size);
    clSetKernelArg(cl_ctx.init_kernel, 10, sizeof(cl_int), &num_leaders);

    size_t global_work_size = pop_size;
    err = clEnqueueNDRangeKernel(cl_ctx.cl_ctx, cl_ctx.init_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
    check_cl_error(err, "clEnqueueNDRangeKernel initialize_population");
    clFinish(cl_ctx.cl_ctx);

    cl_ulong start, end;
    if (init_event) {
        err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.init_time += (end - start) / 1e6;
            }
        }
        clReleaseEvent(init_event);
    }

    // Evaluate initial population and initialize best solution
    cl_event read_event, write_event;
    float *positions = (float *)malloc(pop_size * dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.cl_ctx, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer population");
    float *fitness = (float *)malloc(pop_size * sizeof(float));
    float best_fitness = INFINITY;
    int best_idx = 0;
    for (int i = 0; i < pop_size; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
        fitness[i] = (float)objective_function(pos);
        if (isnan(fitness[i]) || isinf(fitness[i])) fitness[i] = INFINITY;
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            best_idx = i;
        }
        free(pos);
    }
    err = clEnqueueWriteBuffer(cl_ctx.cl_ctx, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer fitness");

    // Initialize best solution
    ctx.best_fitness = best_fitness;
    err = clEnqueueWriteBuffer(cl_ctx.cl_ctx, ctx.best_solution, CL_TRUE, 0, dim * sizeof(cl_float), &positions[best_idx * dim], 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer initial best_solution");
    clFinish(cl_ctx.cl_ctx); // Ensure initial best solution is written

    // Validate best solution
    float *best_pos_validate = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.cl_ctx, ctx.best_solution, CL_TRUE, 0, dim * sizeof(cl_float), best_pos_validate, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueReadBuffer validate best_solution");
    double *pos_validate = (double *)malloc(dim * sizeof(double));
    for (int j = 0; j < dim; j++) pos_validate[j] = best_pos_validate[j];
    float validate_fitness = (float)objective_function(pos_validate);
    if (fabs(validate_fitness - ctx.best_fitness) > 1e-5) {
        fprintf(stderr, "Warning: Initial best_solution fitness mismatch! Expected %f, got %f\n", ctx.best_fitness, validate_fitness);
    }
    printf("Initial best_fitness: %f, best_solution: [", ctx.best_fitness);
    for (int j = 0; j < dim; j++) {
        printf("%f", best_pos_validate[j]);
        if (j < dim - 1) printf(", ");
    }
    printf("]\n");
    free(best_pos_validate);
    free(pos_validate);

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

    // Initialize leaders and pbest on CPU
    float *leaders_pos = (float *)malloc(INITIAL_LEADERS * dim * sizeof(float));
    float *leader_fitness = (float *)malloc(INITIAL_LEADERS * sizeof(float));
    float *pbest_pos = (float *)malloc(pop_size * dim * sizeof(float));
    float *pbest_fitness = (float *)malloc(pop_size * sizeof(float));
    for (int i = 0; i < INITIAL_LEADERS; i++) leader_fitness[i] = INFINITY;
    for (int i = 0; i < pop_size; i++) pbest_fitness[i] = INFINITY;

    // Update leaders and pbest
    for (int i = 0; i < pop_size; i++) {
        if (fitness[i] < pbest_fitness[i]) {
            pbest_fitness[i] = fitness[i];
            for (int j = 0; j < dim; j++) pbest_pos[i * dim + j] = positions[i * dim + j];
        }
        for (int k = 0; k < num_leaders; k++) {
            if (fitness[i] < leader_fitness[k]) {
                for (int l = num_leaders - 1; l > k; l--) {
                    leader_fitness[l] = leader_fitness[l - 1];
                    for (int j = 0; j < dim; j++) leaders_pos[l * dim + j] = leaders_pos[(l - 1) * dim + j];
                }
                leader_fitness[k] = fitness[i];
                for (int j = 0; j < dim; j++) leaders_pos[k * dim + j] = positions[i * dim + j];
                break;
            }
        }
    }

    // Write leaders and pbest to GPU
    err = clEnqueueWriteBuffer(cl_ctx.cl_ctx, ctx.leaders, CL_TRUE, 0, num_leaders * dim * sizeof(cl_float), leaders_pos, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer leaders");
    err = clEnqueueWriteBuffer(cl_ctx.cl_ctx, ctx.leader_fitness, CL_TRUE, 0, num_leaders * sizeof(cl_float), leader_fitness, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer leader_fitness");
    err = clEnqueueWriteBuffer(cl_ctx.cl_ctx, ctx.pbest, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), pbest_pos, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer pbest");
    err = clEnqueueWriteBuffer(cl_ctx.cl_ctx, ctx.pbest_fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), pbest_fitness, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer pbest_fitness");
    clFinish(cl_ctx.cl_ctx); // Ensure buffers are updated

    float *history = (float *)calloc(INITIAL_LEADERS * opt->max_iter, sizeof(float));
    for (int i = 0; i < num_leaders; i++) history[i] = leader_fitness[i];
    err = clEnqueueWriteBuffer(cl_ctx.cl_ctx, leader_fitness_history, CL_TRUE, 0, INITIAL_LEADERS * opt->max_iter * sizeof(cl_float), history, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer leader_fitness_history");

    // Main optimization loop
    while (func_count < max_evals && iteration < opt->max_iter) {
        iteration++;
        printf("Iteration %d\n", iteration);

        // Update positions using GWO
        cl_event update_event;
        cl_float a = A_MAX - (iteration * (A_MAX - A_MIN) / opt->max_iter);
        clSetKernelArg(cl_ctx.update_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.update_kernel, 1, sizeof(cl_mem), &ctx.leaders);
        clSetKernelArg(cl_ctx.update_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.update_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.update_kernel, 4, sizeof(cl_float), &a);
        clSetKernelArg(cl_ctx.update_kernel, 5, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.update_kernel, 6, sizeof(cl_int), &pop_size);
        clSetKernelArg(cl_ctx.update_kernel, 7, sizeof(cl_int), &num_leaders);

        err = clEnqueueNDRangeKernel(cl_ctx.cl_ctx, cl_ctx.update_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &update_event);
        check_cl_error(err, "clEnqueueNDRangeKernel update_positions_gwo");
        clFinish(cl_ctx.cl_ctx);

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

        // Evaluate population
        err = clEnqueueReadBuffer(cl_ctx.cl_ctx, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population");
        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            fitness[i] = (float)objective_function(pos);
            if (isnan(fitness[i]) || isinf(fitness[i])) fitness[i] = INFINITY;
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.cl_ctx, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
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

        // Update leaders, pbest, and best solution on CPU
        best_fitness = ctx.best_fitness;
        best_idx = 0;
        for (int i = 0; i < pop_size; i++) {
            if (fitness[i] < pbest_fitness[i]) {
                pbest_fitness[i] = fitness[i];
                for (int j = 0; j < dim; j++) pbest_pos[i * dim + j] = positions[i * dim + j];
            }
            for (int k = 0; k < num_leaders; k++) {
                if (fitness[i] < leader_fitness[k]) {
                    for (int l = num_leaders - 1; l > k; l--) {
                        leader_fitness[l] = leader_fitness[l - 1];
                        for (int j = 0; j < dim; j++) leaders_pos[l * dim + j] = leaders_pos[(l - 1) * dim + j];
                    }
                    leader_fitness[k] = fitness[i];
                    for (int j = 0; j < dim; j++) leaders_pos[k * dim + j] = positions[i * dim + j];
                    break;
                }
            }
            if (fitness[i] < best_fitness) {
                best_fitness = fitness[i];
                best_idx = i;
            }
        }

        // Update best solution if improved
        if (best_fitness < ctx.best_fitness) {
            ctx.best_fitness = best_fitness;
            err = clEnqueueWriteBuffer(cl_ctx.cl_ctx, ctx.best_solution, CL_TRUE, 0, dim * sizeof(cl_float), &positions[best_idx * dim], 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer best_solution");
            clFinish(cl_ctx.cl_ctx);
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
            // Validate best solution
            float *best_pos_validate = (float *)malloc(dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.cl_ctx, ctx.best_solution, CL_TRUE, 0, dim * sizeof(cl_float), best_pos_validate, 0, NULL, NULL);
            check_cl_error(err, "clEnqueueReadBuffer validate best_solution");
            double *pos_validate = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos_validate[j] = best_pos_validate[j];
            float validate_fitness = (float)objective_function(pos_validate);
            if (fabs(validate_fitness - ctx.best_fitness) > 1e-5) {
                fprintf(stderr, "Warning: Best solution fitness mismatch at iteration %d! Expected %f, got %f\n", iteration, ctx.best_fitness, validate_fitness);
            }
            printf("Updated best_fitness at iteration %d: %f, best_solution: [", iteration, ctx.best_fitness);
            for (int j = 0; j < dim; j++) {
                printf("%f", best_pos_validate[j]);
                if (j < dim - 1) printf(", ");
            }
            printf("]\n");
            free(best_pos_validate);
            free(pos_validate);
        }

        // Update leader fitness history
        for (int i = 0; i < num_leaders; i++) history[iteration * INITIAL_LEADERS + i] = leader_fitness[i];
        err = clEnqueueWriteBuffer(cl_ctx.cl_ctx, leader_fitness_history, CL_TRUE, 0, INITIAL_LEADERS * opt->max_iter * sizeof(cl_float), history, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer leader_fitness_history");

        // Adjust number of leaders (V4)
        float tol_iter = opt->max_iter * TOLERANCE_DEFAULT / 100.0f;
        if (iteration >= tol_iter + 1) {
            float curr_fit = leader_fitness[num_leaders - 1];
            if (isinf(curr_fit)) {
                num_leaders--;
            } else {
                int prev_idx = (num_leaders - 1) + ((iteration - (int)tol_iter) * INITIAL_LEADERS);
                if (prev_idx < INITIAL_LEADERS * opt->max_iter) {
                    float diff = curr_fit - history[prev_idx];
                    if ((diff < 0.0f ? diff > -1e-5f : diff < 1e-5f)) {
                        num_leaders--;
                    }
                }
            }
        }
        num_leaders = num_leaders < 1 ? 1 : num_leaders > INITIAL_LEADERS ? INITIAL_LEADERS : num_leaders;

        // Update GPU buffers
        err = clEnqueueWriteBuffer(cl_ctx.cl_ctx, ctx.leaders, CL_TRUE, 0, num_leaders * dim * sizeof(cl_float), leaders_pos, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer leaders");
        err = clEnqueueWriteBuffer(cl_ctx.cl_ctx, ctx.leader_fitness, CL_TRUE, 0, num_leaders * sizeof(cl_float), leader_fitness, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer leader_fitness");
        err = clEnqueueWriteBuffer(cl_ctx.cl_ctx, ctx.pbest, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), pbest_pos, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer pbest");
        err = clEnqueueWriteBuffer(cl_ctx.cl_ctx, ctx.pbest_fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), pbest_fitness, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer pbest_fitness");
        clFinish(cl_ctx.cl_ctx);

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f, Num Leaders = %d\n", iteration, func_count, ctx.best_fitness, num_leaders);
        printf("Profiling (ms): Update Positions = %.3f\n", update_time);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_time);
    printf("Update Positions: %.3f\n", prof.update_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Update Positions: %.3f\n", prof.update_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Copy best solution to optimizer
    float *best_pos = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.cl_ctx, ctx.best_solution, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer best_solution");
    opt->best_solution.fitness = ctx.best_fitness;
    for (int i = 0; i < dim; i++) opt->best_solution.position[i] = (double)best_pos[i]; // Cast to double for optimizer
    // Validate final best solution
    double *final_pos = (double *)malloc(dim * sizeof(double));
    for (int i = 0; i < dim; i++) final_pos[i] = best_pos[i];
    float final_fitness = (float)objective_function(final_pos);
    if (fabs(final_fitness - ctx.best_fitness) > 1e-5) {
        fprintf(stderr, "Warning: Final best_solution fitness mismatch! Expected %f, got %f\n", ctx.best_fitness, final_fitness);
    }
    printf("Final best_fitness: %f, best_solution: [", ctx.best_fitness);
    for (int i = 0; i < dim; i++) {
        printf("%f", best_pos[i]);
        if (i < dim - 1) printf(", ");
    }
    printf("]\n");
    free(final_pos);
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
    clReleaseMemObject(leader_fitness_history);
    free(fitness);
    free(leaders_pos);
    free(leader_fitness);
    free(pbest_pos);
    free(pbest_fitness);
    free(history);
    free(positions);
    DHLO_cleanup_context(&ctx, &cl_ctx);
    DHLO_cleanup_cl(&cl_ctx);

    printf("Optimization completed. Final Best Fitness: %f\n", opt->best_solution.fitness);
}
