#include "SAO.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
// Error checking utility
void check_cl_error(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error: %s (%d)\n", msg, err);
        exit(1);
    }
}
// Preprocess kernel source (if needed)
char *preprocess_kernel_source(const char *source) {
    size_t len = strlen(source);
    char *processed = (char *)malloc(len + 1);
    strcpy(processed, source);
    return processed;
}
void SAO_init_cl(SAOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_PROFILING_ENABLE,
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
    cl_ctx->brownian_kernel = clCreateKernel(cl_ctx->program, "brownian_motion", &err);
    check_cl_error(err, "clCreateKernel brownian_motion");
    cl_ctx->centroid_kernel = clCreateKernel(cl_ctx->program, "calculate_centroid", &err);
    check_cl_error(err, "clCreateKernel calculate_centroid");
    cl_ctx->exploration_kernel = clCreateKernel(cl_ctx->program, "exploration_phase", &err);
    check_cl_error(err, "clCreateKernel exploration_phase");
    cl_ctx->development_kernel = clCreateKernel(cl_ctx->program, "development_phase", &err);
    check_cl_error(err, "clCreateKernel development_phase");
    cl_ctx->reverse_learning_kernel = clCreateKernel(cl_ctx->program, "random_centroid_reverse_learning", &err);
    check_cl_error(err, "clCreateKernel reverse_learning");
}
void SAO_cleanup_cl(SAOCLContext *cl_ctx) {
    if (cl_ctx->reverse_learning_kernel) clReleaseKernel(cl_ctx->reverse_learning_kernel);
    if (cl_ctx->development_kernel) clReleaseKernel(cl_ctx->development_kernel);
    if (cl_ctx->exploration_kernel) clReleaseKernel(cl_ctx->exploration_kernel);
    if (cl_ctx->centroid_kernel) clReleaseKernel(cl_ctx->centroid_kernel);
    if (cl_ctx->brownian_kernel) clReleaseKernel(cl_ctx->brownian_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}
void SAO_init_context(SAOContext *ctx, Optimizer *opt, SAOCLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;
    int pop_size = opt->population_size;
    int dim = opt->dim;
    // Validate population size and dimension
    if (pop_size <= 0) {
        fprintf(stderr, "Error: Invalid population size (%d)\n", pop_size);
        exit(1);
    }
    if (dim <= 0) {
        fprintf(stderr, "Error: Invalid dimension (%d)\n", dim);
        exit(1);
    }
    printf("Initializing context: pop_size=%d, dim=%d\n", pop_size, dim);
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
    ctx->centroid = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer centroid");
    ctx->elite = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer elite");
    ctx->brownian = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer brownian");
    ctx->qq = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer qq");
    ctx->reverse_pop = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer reverse_pop");
    // Initialize random seeds
    cl_uint *seeds = (cl_uint *)malloc(pop_size * sizeof(cl_uint));
    srand(42);
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
    // Initialize elite and best_position to zero
    float *zero_init = (float *)calloc(dim, sizeof(float));
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->elite, CL_TRUE, 0, dim * sizeof(cl_float), zero_init, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer elite init");
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->best_position, CL_TRUE, 0, dim * sizeof(cl_float), zero_init, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer best_position init");
    free(zero_init);
}
void SAO_cleanup_context(SAOContext *ctx, SAOCLContext *cl_ctx) {
    if (ctx->reverse_pop) clReleaseMemObject(ctx->reverse_pop);
    if (ctx->qq) clReleaseMemObject(ctx->qq);
    if (ctx->brownian) clReleaseMemObject(ctx->brownian);
    if (ctx->elite) clReleaseMemObject(ctx->elite);
    if (ctx->centroid) clReleaseMemObject(ctx->centroid);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}
void SAO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Snow Ablation Optimization...\n");
    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device ||
        !opt->population_buffer || !opt->fitness_buffer ||
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }
    // Read kernel source
    FILE *fp = fopen("SAO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open SAO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: SAO.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read SAO.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: SAO.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }
    // Initialize OpenCL context
    SAOCLContext cl_ctx = {0};
    SAO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);
    // Initialize algorithm context
    SAOContext ctx = {0};
    SAO_init_context(&ctx, opt, &cl_ctx);
    int pop_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;
    int num_a = pop_size / 2;
    int num_b = pop_size - num_a;
    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * pop_size : SAO_MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);
    // Profiling data
    typedef struct {
        double init_pop_time;
        double brownian_time;
        double centroid_time;
        double exploration_time;
        double development_time;
        double reverse_learning_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};
    cl_int err;
    cl_mem best_fitness_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_fitness");
    // Initialize population
    printf("Executing initialize_population kernel\n");
    cl_event init_event;
    clSetKernelArg(cl_ctx->init_pop_kernel, 0, sizeof(cl_mem), &ctx.population);
    clSetKernelArg(cl_ctx->init_pop_kernel, 1, sizeof(cl_mem), &ctx.fitness);
    clSetKernelArg(cl_ctx->init_pop_kernel, 2, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx->init_pop_kernel, 3, sizeof(cl_mem), &ctx.best_position);
    clSetKernelArg(cl_ctx->init_pop_kernel, 4, sizeof(cl_mem), &best_fitness_buf);
    clSetKernelArg(cl_ctx->init_pop_kernel, 5, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx->init_pop_kernel, 6, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx->init_pop_kernel, 7, sizeof(cl_int), &pop_size);
    size_t global_work_size = pop_size;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx->init_pop_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
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
    printf("Evaluating initial population\n");
    cl_event read_event, write_event;
    float *positions = (float *)malloc(pop_size * dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx->queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueReadBuffer population");
    float *fitness = (float *)malloc(pop_size * sizeof(float));
    for (int i = 0; i < pop_size; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            pos[j] = positions[i * dim + j];
            if (!isfinite(pos[j])) {
                fprintf(stderr, "Error: Invalid position[%d][%d] = %f\n", i, j, pos[j]);
                exit(1);
            }
        }
        fitness[i] = (float)objective_function(pos);
        if (!isfinite(fitness[i])) {
            fprintf(stderr, "Error: Invalid fitness[%d] = %f\n", i, fitness[i]);
            exit(1);
        }
        free(pos);
    }
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer fitness");
    clFinish(cl_ctx.queue);
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
    // Initialize elite and best position
    printf("Initializing elite and best position\n");
    positions = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx->queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer best_pos");
    for (int i = 0; i < dim; i++) {
        if (!isfinite(positions[i])) {
            fprintf(stderr, "Error: Invalid best_pos[%d] = %f\n", i, positions[i]);
            exit(1);
        }
    }
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(float), positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer best_pos");
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx.elite, CL_TRUE, 0, dim * sizeof(float), positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer elite");
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
    err = clEnqueueWriteBuffer(cl_ctx->queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &best_fitness, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer best_fitness");
    // Main optimization loop
    while (func_count < max_evals) {
        iteration++;
        printf("Iteration %d\n", iteration);
        // Calculate snow ablation rate
        float t = (float)iteration / SAO_MAX_ITER;
        float T = exp(-t);
        float denom = SAO_EULER - 1.0f;
        if (fabs(denom) < 1e-6f) {
            fprintf(stderr, "Error: Invalid denominator in Df calculation\n");
            exit(1);
        }
        float Df = SAO_DF_MIN + (SAO_DF_MAX - SAO_DF_MIN) * (exp(t) - 1.0f) / denom;
        float R = Df * T;
        if (!isfinite(R) || !isfinite(T) || !isfinite(Df)) {
            fprintf(stderr, "Error: Invalid snow ablation parameters: R=%f, T=%f, Df=%f\n", R, T, Df);
            exit(1);
        }
        printf("Snow ablation: t=%f, T=%f, Df=%f, R=%f\n", t, T, Df, R);
        // Brownian motion
        printf("Executing brownian_motion kernel\n");
        cl_event brownian_event;
        clSetKernelArg(cl_ctx->brownian_kernel, 0, sizeof(cl_mem), &ctx.brownian);
        clSetKernelArg(cl_ctx->brownian_kernel, 1, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx->brownian_kernel, 2, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx->brownian_kernel, 3, sizeof(cl_int), &num_a);
        global_work_size = num_a;
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx->brownian_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &brownian_event);
        check_cl_error(err, "clEnqueueNDRangeKernel brownian_motion");
        clFinish(cl_ctx.queue);
        if (brownian_event) {
            err = clGetEventProfilingInfo(brownian_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(brownian_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.brownian_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(brownian_event);
        }
        // Check brownian values
        float *brownian_check = (float *)malloc(num_a * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx->queue, ctx.brownian, CL_TRUE, 0, num_a * dim * sizeof(cl_float), brownian_check, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer brownian_check");
        for (int i = 0; i < num_a * dim; i++) {
            if (!isfinite(brownian_check[i])) {
                fprintf(stderr, "Error: Invalid brownian[%d] = %f\n", i, brownian_check[i]);
            }
            (1);
        }
        free(brownian_check);
        // Calculate centroid
        printf("Executing calculate_centroid kernel\n");
        cl_event centroid_event;
        clSetKernelArg(cl_ctx->centroid_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx->centroid_kernel, 1, sizeof(cl_mem), &ctx.centroid);
        clSetKernelArg(cl_ctx->centroid_kernel, 2, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx->centroid_kernel, 3, sizeof(cl_int), &pop_size);
        global_work_size = dim;
        err = clEnqueueNDRangeKernel(cl_ctx->queue, cl_ctx->centroid_kernel, 1,  NULL, &global_work_size, NULL, 0, NULL, &centroid_event);
        check_cl_error(err, "clEnqueueNDRangeKernel calculate_centroid"));
        clFinish(cl_ctx->queue);
        if (centroid_event) {
            err = clGetEventProfilingInfo(centroid_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(cl_centroid_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.centroid_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(centroid_event);
        }
        // Check centroid values
        float *centroid_check = (float *)malloc(dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx->queue, ctx.centroid, CL_TRUE, 0, dim * sizeof(cl_float), centroid_check, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer centroid_check");
        for (int i = 0; i < dim; i++) {
            if (!isfinite(centroid_check[i])) {
                fprintf(stderr, "Error: Invalid centroid[%d] = %f\n", i, centroid_check[i]);
                exit(1);
            }
        }
        free(centroid_check);
        // Exploration phase
        printf("Executing exploration_phase kernel\n");
        cl_event exploration_event;
        clSetKernelArg(cl_ctx->exploration_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx->exploration_kernel, 1, sizeof(cl_mem), &ctx.brownian);
        clSetKernelArg(cl_ctx->exploration_kernel, 2, sizeof(cl_mem), &ctx.centroid);
        clSetKernelArg(cl_ctx->exploration_kernel, 3, sizeof(cl_mem), &ctx.elite);
        clSetKernelArg(cl_ctx->exploration_kernel, 4, sizeof(cl_mem), &ctx.best_position);
        clSetKernelArg(cl_ctx->exploration_kernel, 5, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx->exploration_kernel, 6, sizeof(cl_float), &R);
        clSetKernelArg(cl_ctx->exploration_kernel, 7, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx->exploration_kernel, 8, sizeof(cl_int), &num_a);
        global_work_size = num_a;
        err = clEnqueueNDRangeKernel(cl_ctx->queue, cl_ctx->exploration_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &exploration_event);
        check_cl_error(err, "clEnqueueNDRangeKernel exploration_phase");
        clFinish(cl_ctx->queue);
        if (exploration_event) {
            err = clGetEventProfilingInfo(exploration_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(exploration_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.exploration_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(exploration_event);
        }
        // Development phase
        printf("Executing development_phase kernel\n");
        cl_event development_event;
        clSetKernelArg(cl_ctx->development_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx->development_kernel, 1, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx->development_kernel, 2, sizeof(cl_float), &R);
        clSetKernelArg(cl_ctx->development_kernel, 3, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx->development_kernel, 4, sizeof(cl_int), &num_a);
        clSetKernelArg(cl_ctx->development_kernel, 5, sizeof(cl_int), &pop_size);
        global_work_size = pop_size;
        err = clEnqueueNDRangeKernel(cl_ctx->queue, cl_ctx->development_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &development_event);
        check_cl_error(err, "clEnqueueNDRangeKernel development_phase");
        clFinish(cl_ctx->queue);
        if (development_event) {
            err = clGetEventProfilingInfo(development_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(development_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.development_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(development_event);
        }
        // Evaluate population on CPU
        printf("Evaluating population\n");
        positions = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx->queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population");
        float *reverse_fitness = (float *)malloc(pop_size * sizeof(float));
        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) {
                pos[j] = positions[i * dim + j];
                if (!isfinite(pos[j])) {
                    fprintf(stderr, "Error: Invalid position[%d][%d] = %f\n", i, j, pos[j]);
                    exit(1);
                }
            }
            fitness[i] = (float)objective_function(pos);
            if (!isfinite(fitness[i])) {
                fprintf(stderr, "Error: Invalid fitness[%d] = %f\n", i, fitness[i]);
                exit(1);
            }
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx->queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer fitness");
        clFinish(cl_ctx.queue);
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
        // Random centroid reverse learning
        printf("Executing random_centroid_reverse_learning kernel\n");
        int B = 2 + (rand() % ((pop_size / 2) - 1));
        if (B < 1) B = 1;
        printf("Reverse learning: B=%d\n", B);
        cl_event reverse_learning_event;
        clSetKernelArg(cl_ctx->reverse_learning_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx->reverse_learning_kernel, 1, sizeof(cl_mem), &ctx.reverse_pop);
        clSetKernelArg(cl_ctx->reverse_learning_kernel, 2, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx->reverse_learning_kernel, 3, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx->reverse_learning_kernel, 4, sizeof(cl_int), &pop_size);
        clSetKernelArg(cl_ctx->reverse_learning_kernel, 5, sizeof(cl_int), &B);
        global_work_size = pop_size;
        err = clEnqueueNDRangeKernel(cl_ctx->queue, cl_ctx->reverse_learning_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &reverse_learning_event);
        check_cl_error(err, "clEnqueueNDRangeKernel reverse_learning");
        clFinish(cl_ctx->queue);
        if (reverse_learning_event) {
            err = clGetEventProfilingInfo(reverse_learning_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(reverse_learning_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.reverse_learning_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(reverse_learning_event);
        }
        // Evaluate reverse population
        printf("Evaluating reverse population\n");
        positions = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx->queue, ctx.reverse_pop, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer reverse_pop");
        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) {
                pos[j] = positions[i * dim + j];
                if (!isfinite(pos[j])) {
                    fprintf(stderr, "Error: Invalid reverse_pop[%d][%d] = %f\n", i, j, pos[j]);
                    exit(1);
                }
            }
            reverse_fitness[i] = (float)objective_function(pos);
            if (!isfinite(reverse_fitness[i])) {
                fprintf(stderr, "Error: Invalid reverse_fitness[%d] = %f\n", i, reverse_fitness[i]);
                exit(1);
            }
            free(pos);
        }
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
        func_count += pop_size;
        // Greedy selection and elite update
        float new_best = fitness[0];
        int best_idx = 0;
        for (int i = 0; i < pop_size; i++) {
            if (reverse_fitness[i] < fitness[i]) {
                positions = (float *)malloc(dim * sizeof(float));
                err = clEnqueueReadBuffer(cl_ctx->queue, ctx.reverse_pop, CL_TRUE, i * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
                check_cl_error(err, "clEnqueueReadBuffer reverse_pop");
                for (int j = 0; j < dim; j++) {
                    if (!isfinite(positions[j])) {
                        fprintf(stderr, "Error: Invalid reverse_pop[%d][%d] = %f\n", i, j, positions[j]);
                        exit(1);
                    }
                }
                err = clEnqueueWriteBuffer(cl_ctx->queue, ctx.population, CL_TRUE, i * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &write_event);
                check_cl_error(err, "clEnqueueWriteBuffer population");
                fitness[i] = reverse_fitness[i];
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
            if (fitness[i] < new_best) {
                new_best = fitness[i];
                best_idx = i;
            }
        }
        free(reverse_fitness);
        if (new_best < ctx.best_fitness) {
            ctx.best_fitness = new_best;
            positions = (float *)malloc(dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx->queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer best_pos");
            for (int i = 0; i < dim; i++) {
                if (!isfinite(positions[i])) {
                    fprintf(stderr, "Error: Invalid best_pos[%d] = %f\n", i, positions[i]);
                    exit(1);
                }
            }
            err = clEnqueueWriteBuffer(cl_ctx->queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer best_pos");
            err = clEnqueueWriteBuffer(cl_ctx->queue, ctx.elite, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer elite");
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
            err = clEnqueueWriteBuffer(cl_ctx->queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &new_best, 0, NULL, &write_event);
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
        // Adjust subpopulation sizes
        if (num_a < pop_size) {
            num_a += 1;
            num_b--;
        }
        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.best_fitness);
    }
    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_pop_time);
    printf("Brownian Motion: %.3f\n", prof.brownian_time);
    printf("Centroid Calculation: %.3f\n", prof.centroid_time);
    printf("Exploration Phase: %.3f\n", prof.exploration_time);
    printf("Development Phase: %.3f\n", prof.development_time);
    printf("Reverse Learning: %.3f\n", prof.reverse_learning_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Brownian Motion: %.3f\n", prof.brownian_time / prof.count);
        printf("Centroid Calculation: %.3f\n", prof.centroid_time / prof.count);
        printf("Exploration Phase: %.3f\n", prof.exploration_time / prof.count);
        printf("Development Phase: %.3f\n", prof.development_time / prof.count);
        printf("Reverse Learning: %.3f\n", prof.reverse_learning_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }
    // Copy best solution to optimizer
    float *best_pos = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer final best_pos");
    for (int i = 0; i < dim; i++) {
        if (!isfinite(best_pos[i])) {
            fprintf(stderr, "Error: Invalid final best_pos[%d] = %f\n", i, best_pos[i]);
            exit(1);
        }
    }
    opt->best_solution.fitness = ctx.best_fitness;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = best_pos[i];
    }
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
    clReleaseMemObject(best_fitness_buf);
    SAO_cleanup_context(&ctx, &cl_ctx);
    SAO_cleanup_cl(&cl_ctx);
    printf("Optimization completed\n");
}
