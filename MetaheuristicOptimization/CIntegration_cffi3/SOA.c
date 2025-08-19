#include "SOA.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (stub, can be expanded if needed)
char *preprocess_kernel_source(const char *source);

void SOA_init_cl(SOACLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->update_kernel = clCreateKernel(cl_ctx->program, "update_solutions", &err);
    check_cl_error(err, "clCreateKernel update_solutions");
}

void SOA_cleanup_cl(SOACLContext *cl_ctx) {
    if (cl_ctx->update_kernel) clReleaseKernel(cl_ctx->update_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void SOA_init_context(SOAContext *ctx, Optimizer *opt, SOACLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int num_regions = NUM_REGIONS;

    // Allocate OpenCL buffers
    ctx->population = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer population");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->pbest_s = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer pbest_s");
    ctx->pbest_fun = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer pbest_fun");
    ctx->lbest_s = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, num_regions * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer lbest_s");
    ctx->lbest_fun = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, num_regions * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer lbest_fun");
    ctx->e_t_1 = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer e_t_1");
    ctx->e_t_2 = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer e_t_2");
    ctx->f_t_1 = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer f_t_1");
    ctx->f_t_2 = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer f_t_2");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->start_reg = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, num_regions * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer start_reg");
    ctx->end_reg = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, num_regions * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer end_reg");
    ctx->size_reg = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, num_regions * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer size_reg");
    ctx->rmax = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer rmax");
    ctx->rmin = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer rmin");
    ctx->best_position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_position");

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

    // Initialize regions
    int *start_reg = (int *)malloc(num_regions * sizeof(int));
    int *end_reg = (int *)malloc(num_regions * sizeof(int));
    int *size_reg = (int *)malloc(num_regions * sizeof(int));
    for (int r = 0; r < num_regions; r++) {
        start_reg[r] = (r * pop_size) / num_regions;
        end_reg[r] = ((r + 1) * pop_size) / num_regions;
        size_reg[r] = end_reg[r] - start_reg[r];
    }
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->start_reg, CL_TRUE, 0, num_regions * sizeof(cl_int), start_reg, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer start_reg");
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->end_reg, CL_TRUE, 0, num_regions * sizeof(cl_int), end_reg, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer end_reg");
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->size_reg, CL_TRUE, 0, num_regions * sizeof(cl_int), size_reg, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer size_reg");
    free(start_reg);
    free(end_reg);
    free(size_reg);

    // Initialize step sizes
    float *rmax = (float *)malloc(dim * sizeof(float));
    float *rmin = (float *)malloc(dim * sizeof(float));
    for (int j = 0; j < dim; j++) {
        rmax[j] = 0.5f * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        rmin[j] = -rmax[j];
    }
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->rmax, CL_TRUE, 0, dim * sizeof(cl_float), rmax, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer rmax");
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->rmin, CL_TRUE, 0, dim * sizeof(cl_float), rmin, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer rmin");
    free(rmax);
    free(rmin);
}

void SOA_cleanup_context(SOAContext *ctx, SOACLContext *cl_ctx) {
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->rmin) clReleaseMemObject(ctx->rmin);
    if (ctx->rmax) clReleaseMemObject(ctx->rmax);
    if (ctx->size_reg) clReleaseMemObject(ctx->size_reg);
    if (ctx->end_reg) clReleaseMemObject(ctx->end_reg);
    if (ctx->start_reg) clReleaseMemObject(ctx->start_reg);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->f_t_2) clReleaseMemObject(ctx->f_t_2);
    if (ctx->f_t_1) clReleaseMemObject(ctx->f_t_1);
    if (ctx->e_t_2) clReleaseMemObject(ctx->e_t_2);
    if (ctx->e_t_1) clReleaseMemObject(ctx->e_t_1);
    if (ctx->lbest_fun) clReleaseMemObject(ctx->lbest_fun);
    if (ctx->lbest_s) clReleaseMemObject(ctx->lbest_s);
    if (ctx->pbest_fun) clReleaseMemObject(ctx->pbest_fun);
    if (ctx->pbest_s) clReleaseMemObject(ctx->pbest_s);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

void SOA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU SOA optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device ||
        !opt->population_buffer || !opt->fitness_buffer ||
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("SOA.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open SOA.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: SOA.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read SOA.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: SOA.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    SOACLContext cl_ctx = {0};
    SOA_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    SOAContext ctx = {0};
    SOA_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int num_regions = NUM_REGIONS;
    int func_count = 0;
    int iteration = 0;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * pop_size : SOA_MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_pop_time;
        double update_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;
    cl_mem best_fitness_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_fitness");

    // Allocate reusable arrays
    float *positions = (float *)malloc(pop_size * dim * sizeof(float));
    if (!positions) {
        fprintf(stderr, "Error: Memory allocation failed for positions\n");
        exit(1);
    }
    double *fitness = (double *)malloc(pop_size * sizeof(double));
    if (!fitness) {
        fprintf(stderr, "Error: Memory allocation failed for fitness\n");
        free(positions);
        exit(1);
    }
    float *best_pos = (float *)malloc(dim * sizeof(float));
    if (!best_pos) {
        fprintf(stderr, "Error: Memory allocation failed for best_pos\n");
        free(positions);
        free(fitness);
        exit(1);
    }

    // Initialize population
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_pop_kernel, 0, sizeof(cl_mem), &ctx.population);
    clSetKernelArg(cl_ctx.init_pop_kernel, 1, sizeof(cl_mem), &ctx.fitness);
    clSetKernelArg(cl_ctx.init_pop_kernel, 2, sizeof(cl_mem), &ctx.pbest_s);
    clSetKernelArg(cl_ctx.init_pop_kernel, 3, sizeof(cl_mem), &ctx.pbest_fun);
    clSetKernelArg(cl_ctx.init_pop_kernel, 4, sizeof(cl_mem), &ctx.e_t_1);
    clSetKernelArg(cl_ctx.init_pop_kernel, 5, sizeof(cl_mem), &ctx.e_t_2);
    clSetKernelArg(cl_ctx.init_pop_kernel, 6, sizeof(cl_mem), &ctx.f_t_1);
    clSetKernelArg(cl_ctx.init_pop_kernel, 7, sizeof(cl_mem), &ctx.f_t_2);
    clSetKernelArg(cl_ctx.init_pop_kernel, 8, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 9, sizeof(cl_mem), &ctx.best_position);
    clSetKernelArg(cl_ctx.init_pop_kernel, 10, sizeof(cl_mem), &best_fitness_buf);
    clSetKernelArg(cl_ctx.init_pop_kernel, 11, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 12, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_pop_kernel, 13, sizeof(cl_int), &pop_size);

    size_t global_work_size = pop_size;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.init_pop_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
    check_cl_error(err, "clEnqueueNDRangeKernel init_pop");
    clFinish(cl_ctx.queue);

    cl_ulong start, end;
    if (init_event) {
        err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Warning: clGetEventProfilingInfo init_pop start failed: %d\n", err);
        } else {
            err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Warning: clGetEventProfilingInfo init_pop end failed: %d\n", err);
            } else {
                prof.init_pop_time += (end - start) / 1e6; // Convert ns to ms
            }
        }
        clReleaseEvent(init_event);
    }

    // Initialize local bests on CPU
    float *lbest_s = (float *)malloc(num_regions * dim * sizeof(float));
    if (!lbest_s) {
        fprintf(stderr, "Error: Memory allocation failed for lbest_s\n");
        free(positions);
        free(fitness);
        free(best_pos);
        exit(1);
    }
    float *lbest_fun = (float *)malloc(num_regions * sizeof(float));
    if (!lbest_fun) {
        fprintf(stderr, "Error: Memory allocation failed for lbest_fun\n");
        free(positions);
        free(fitness);
        free(best_pos);
        free(lbest_s);
        exit(1);
    }
    for (int r = 0; r < num_regions; r++) {
        for (int j = 0; j < dim; j++) {
            lbest_s[r * dim + j] = 0.0f;
        }
        lbest_fun[r] = INFINITY;
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.lbest_s, CL_TRUE, 0, num_regions * dim * sizeof(cl_float), lbest_s, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer lbest_s");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.lbest_fun, CL_TRUE, 0, num_regions * sizeof(cl_float), lbest_fun, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer lbest_fun");
    free(lbest_s);
    free(lbest_fun);

    // Evaluate initial population on CPU
    cl_event read_event, write_event;
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer population");
    for (int i = 0; i < pop_size; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        if (!pos) {
            fprintf(stderr, "Error: Memory allocation failed for pos\n");
            free(positions);
            free(fitness);
            free(best_pos);
            exit(1);
        }
        for (int j = 0; j < dim; j++) pos[j] = (double)positions[i * dim + j];
        fitness[i] = objective_function(pos);
        if (isnan(fitness[i]) || isinf(fitness[i])) {
            fprintf(stderr, "Error: Invalid fitness value (NaN or Inf) at index %d\n", i);
            free(pos);
            free(positions);
            free(fitness);
            free(best_pos);
            exit(1);
        }
        free(pos);
    }
    float *fitness_float = (float *)malloc(pop_size * sizeof(float));
    if (!fitness_float) {
        fprintf(stderr, "Error: Memory allocation failed for fitness_float\n");
        free(positions);
        free(fitness);
        free(best_pos);
        exit(1);
    }
    for (int i = 0; i < pop_size; i++) {
        fitness_float[i] = (float)fitness[i];
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness_float, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer fitness");
    free(fitness_float);

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
    double best_fitness = fitness[0];
    int best_idx = 0;
    for (int i = 1; i < pop_size; i++) {
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            best_idx = i;
        }
    }
    ctx.best_fitness = (float)best_fitness;

    for (int j = 0; j < dim; j++) {
        best_pos[j] = positions[best_idx * dim + j];
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &write_event);
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

    float best_fitness_float = (float)best_fitness;
    err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &best_fitness_float, 0, NULL, &write_event);
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

    // Update local bests on CPU
    int *start_reg = (int *)malloc(num_regions * sizeof(int));
    if (!start_reg) {
        fprintf(stderr, "Error: Memory allocation failed for start_reg\n");
        free(positions);
        free(fitness);
        free(best_pos);
        exit(1);
    }
    int *end_reg = (int *)malloc(num_regions * sizeof(int));
    if (!end_reg) {
        fprintf(stderr, "Error: Memory allocation failed for end_reg\n");
        free(positions);
        free(fitness);
        free(best_pos);
        free(start_reg);
        exit(1);
    }
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.start_reg, CL_TRUE, 0, num_regions * sizeof(cl_int), start_reg, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueReadBuffer start_reg");
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.end_reg, CL_TRUE, 0, num_regions * sizeof(cl_int), end_reg, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueReadBuffer end_reg");
    float *new_lbest_s = (float *)malloc(num_regions * dim * sizeof(float));
    if (!new_lbest_s) {
        fprintf(stderr, "Error: Memory allocation failed for new_lbest_s\n");
        free(positions);
        free(fitness);
        free(best_pos);
        free(start_reg);
        free(end_reg);
        exit(1);
    }
    float *new_lbest_fun = (float *)malloc(num_regions * sizeof(float));
    if (!new_lbest_fun) {
        fprintf(stderr, "Error: Memory allocation failed for new_lbest_fun\n");
        free(positions);
        free(fitness);
        free(best_pos);
        free(start_reg);
        free(end_reg);
        free(new_lbest_s);
        exit(1);
    }
    for (int r = 0; r < num_regions; r++) {
        int best_idx = start_reg[r];
        double best_fun = fitness[best_idx];
        for (int i = start_reg[r]; i < end_reg[r]; i++) {
            if (fitness[i] < best_fun) {
                best_fun = fitness[i];
                best_idx = i;
            }
        }
        new_lbest_fun[r] = (float)best_fun;
        for (int j = 0; j < dim; j++) {
            new_lbest_s[r * dim + j] = positions[best_idx * dim + j];
        }
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.lbest_s, CL_TRUE, 0, num_regions * dim * sizeof(cl_float), new_lbest_s, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer lbest_s");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.lbest_fun, CL_TRUE, 0, num_regions * sizeof(cl_float), new_lbest_fun, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer lbest_fun");
    free(start_reg);
    free(end_reg);
    free(new_lbest_s);
    free(new_lbest_fun);

    // Main optimization loop
    while (func_count < max_evals) {
        iteration++;
        printf("Iteration %d\n", iteration);

        // Update solutions
        cl_event update_event;
        cl_float weight = W_MAX_SOA - (float)iteration * (W_MAX_SOA - W_MIN_SOA) / opt->max_iter;
        cl_float mu = SOA_MU_MAX - (float)iteration * (SOA_MU_MAX - SOA_MU_MIN) / opt->max_iter;
        clSetKernelArg(cl_ctx.update_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.update_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.update_kernel, 2, sizeof(cl_mem), &ctx.pbest_s);
        clSetKernelArg(cl_ctx.update_kernel, 3, sizeof(cl_mem), &ctx.pbest_fun);
        clSetKernelArg(cl_ctx.update_kernel, 4, sizeof(cl_mem), &ctx.lbest_s);
        clSetKernelArg(cl_ctx.update_kernel, 5, sizeof(cl_mem), &ctx.lbest_fun);
        clSetKernelArg(cl_ctx.update_kernel, 6, sizeof(cl_mem), &ctx.e_t_1);
        clSetKernelArg(cl_ctx.update_kernel, 7, sizeof(cl_mem), &ctx.e_t_2);
        clSetKernelArg(cl_ctx.update_kernel, 8, sizeof(cl_mem), &ctx.f_t_1);
        clSetKernelArg(cl_ctx.update_kernel, 9, sizeof(cl_mem), &ctx.f_t_2);
        clSetKernelArg(cl_ctx.update_kernel, 10, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.update_kernel, 11, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.update_kernel, 12, sizeof(cl_mem), &ctx.start_reg);
        clSetKernelArg(cl_ctx.update_kernel, 13, sizeof(cl_mem), &ctx.end_reg);
        clSetKernelArg(cl_ctx.update_kernel, 14, sizeof(cl_mem), &ctx.size_reg);
        clSetKernelArg(cl_ctx.update_kernel, 15, sizeof(cl_mem), &ctx.rmax);
        clSetKernelArg(cl_ctx.update_kernel, 16, sizeof(cl_mem), &ctx.rmin);
        clSetKernelArg(cl_ctx.update_kernel, 17, sizeof(cl_float), &weight);
        clSetKernelArg(cl_ctx.update_kernel, 18, sizeof(cl_float), &mu);
        clSetKernelArg(cl_ctx.update_kernel, 19, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.update_kernel, 20, sizeof(cl_int), &pop_size);
        clSetKernelArg(cl_ctx.update_kernel, 21, sizeof(cl_int), &num_regions);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &update_event);
        check_cl_error(err, "clEnqueueNDRangeKernel update_solutions");
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
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population");
        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            if (!pos) {
                fprintf(stderr, "Error: Memory allocation failed for pos\n");
                free(positions);
                free(fitness);
                free(best_pos);
                exit(1);
            }
            for (int j = 0; j < dim; j++) pos[j] = (double)positions[i * dim + j];
            fitness[i] = objective_function(pos);
            if (isnan(fitness[i]) || isinf(fitness[i])) {
                fprintf(stderr, "Error: Invalid fitness value (NaN or Inf) at index %d, iteration %d\n", i, iteration);
                free(pos);
                free(positions);
                free(fitness);
                free(best_pos);
                exit(1);
            }
            free(pos);
        }
        float *fitness_float = (float *)malloc(pop_size * sizeof(float));
        if (!fitness_float) {
            fprintf(stderr, "Error: Memory allocation failed for fitness_float\n");
            free(positions);
            free(fitness);
            free(best_pos);
            exit(1);
        }
        for (int i = 0; i < pop_size; i++) {
            fitness_float[i] = (float)fitness[i];
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness_float, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer fitness");
        free(fitness_float);

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

        // Update personal bests on CPU
        float *pbest_s = (float *)malloc(pop_size * dim * sizeof(float));
        if (!pbest_s) {
            fprintf(stderr, "Error: Memory allocation failed for pbest_s\n");
            free(positions);
            free(fitness);
            free(best_pos);
            exit(1);
        }
        float *pbest_fun = (float *)malloc(pop_size * sizeof(float));
        if (!pbest_fun) {
            fprintf(stderr, "Error: Memory allocation failed for pbest_fun\n");
            free(positions);
            free(fitness);
            free(best_pos);
            free(pbest_s);
            exit(1);
        }
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.pbest_s, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), pbest_s, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer pbest_s");
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.pbest_fun, CL_TRUE, 0, pop_size * sizeof(cl_float), pbest_fun, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer pbest_fun");
        for (int i = 0; i < pop_size; i++) {
            if (fitness[i] < (double)pbest_fun[i]) {
                pbest_fun[i] = (float)fitness[i];
                for (int j = 0; j < dim; j++) {
                    pbest_s[i * dim + j] = positions[i * dim + j];
                }
            }
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.pbest_s, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), pbest_s, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer pbest_s");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.pbest_fun, CL_TRUE, 0, pop_size * sizeof(cl_float), pbest_fun, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer pbest_fun");
        free(pbest_s);
        free(pbest_fun);

        // Update local bests on CPU
        start_reg = (int *)malloc(num_regions * sizeof(int));
        if (!start_reg) {
            fprintf(stderr, "Error: Memory allocation failed for start_reg\n");
            free(positions);
            free(fitness);
            free(best_pos);
            exit(1);
        }
        end_reg = (int *)malloc(num_regions * sizeof(int));
        if (!end_reg) {
            fprintf(stderr, "Error: Memory allocation failed for end_reg\n");
            free(positions);
            free(fitness);
            free(best_pos);
            free(start_reg);
            exit(1);
        }
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.start_reg, CL_TRUE, 0, num_regions * sizeof(cl_int), start_reg, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer start_reg");
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.end_reg, CL_TRUE, 0, num_regions * sizeof(cl_int), end_reg, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer end_reg");
        new_lbest_s = (float *)malloc(num_regions * dim * sizeof(float));
        if (!new_lbest_s) {
            fprintf(stderr, "Error: Memory allocation failed for new_lbest_s\n");
            free(positions);
            free(fitness);
            free(best_pos);
            free(start_reg);
            free(end_reg);
            exit(1);
        }
        new_lbest_fun = (float *)malloc(num_regions * sizeof(float));
        if (!new_lbest_fun) {
            fprintf(stderr, "Error: Memory allocation failed for new_lbest_fun\n");
            free(positions);
            free(fitness);
            free(best_pos);
            free(start_reg);
            free(end_reg);
            free(new_lbest_s);
            exit(1);
        }
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.lbest_s, CL_TRUE, 0, num_regions * dim * sizeof(cl_float), new_lbest_s, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer lbest_s");
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.lbest_fun, CL_TRUE, 0, num_regions * sizeof(cl_float), new_lbest_fun, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer lbest_fun");
        for (int r = 0; r < num_regions; r++) {
            int best_idx = start_reg[r];
            double best_fun = fitness[best_idx];
            for (int i = start_reg[r]; i < end_reg[r]; i++) {
                if (fitness[i] < best_fun) {
                    best_fun = fitness[i];
                    best_idx = i;
                }
            }
            if (best_fun < (double)new_lbest_fun[r]) {
                new_lbest_fun[r] = (float)best_fun;
                for (int j = 0; j < dim; j++) {
                    new_lbest_s[r * dim + j] = positions[best_idx * dim + j];
                }
            }
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.lbest_s, CL_TRUE, 0, num_regions * dim * sizeof(cl_float), new_lbest_s, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer lbest_s");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.lbest_fun, CL_TRUE, 0, num_regions * sizeof(cl_float), new_lbest_fun, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer lbest_fun");
        free(start_reg);
        free(end_reg);
        free(new_lbest_s);
        free(new_lbest_fun);

        // Update global best
        double new_best = fitness[0];
        best_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (fitness[i] < new_best) {
                new_best = fitness[i];
                best_idx = i;
            }
        }
        if (new_best < (double)ctx.best_fitness) {
            ctx.best_fitness = (float)new_best;
            for (int j = 0; j < dim; j++) {
                best_pos[j] = positions[best_idx * dim + j];
            }
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &write_event);
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

            best_fitness_float = (float)new_best;
            err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &best_fitness_float, 0, NULL, &write_event);
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

        // Update history
        float *temp_e_t_2 = (float *)malloc(pop_size * dim * sizeof(float));
        if (!temp_e_t_2) {
            fprintf(stderr, "Error: Memory allocation failed for temp_e_t_2\n");
            free(positions);
            free(fitness);
            free(best_pos);
            exit(1);
        }
        float *temp_f_t_2 = (float *)malloc(pop_size * sizeof(float));
        if (!temp_f_t_2) {
            fprintf(stderr, "Error: Memory allocation failed for temp_f_t_2\n");
            free(positions);
            free(fitness);
            free(best_pos);
            free(temp_e_t_2);
            exit(1);
        }
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.e_t_1, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), temp_e_t_2, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer e_t_1");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.e_t_2, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), temp_e_t_2, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer e_t_2");
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), temp_e_t_2, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer population");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.e_t_1, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), temp_e_t_2, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer e_t_1");
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.f_t_1, CL_TRUE, 0, pop_size * sizeof(cl_float), temp_f_t_2, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer f_t_1");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.f_t_2, CL_TRUE, 0, pop_size * sizeof(cl_float), temp_f_t_2, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer f_t_2");
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), temp_f_t_2, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer fitness");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.f_t_1, CL_TRUE, 0, pop_size * sizeof(cl_float), temp_f_t_2, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer f_t_1");
        free(temp_e_t_2);
        free(temp_f_t_2);

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, (double)ctx.best_fitness);
        printf("Profiling (ms): Update Solutions = %.3f\n", update_time);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_pop_time);
    printf("Update Solutions: %.3f\n", prof.update_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Update Solutions: %.3f\n", prof.update_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Copy best solution to optimizer
    float *final_best_pos = (float *)malloc(dim * sizeof(float));
    if (!final_best_pos) {
        fprintf(stderr, "Error: Memory allocation failed for final_best_pos\n");
        free(positions);
        free(fitness);
        free(best_pos);
        exit(1);
    }
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), final_best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer final best_pos");
    opt->best_solution.fitness = (double)ctx.best_fitness;
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = (double)final_best_pos[i];
    }
    free(final_best_pos);
    free(positions);
    free(fitness);
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
    SOA_cleanup_context(&ctx, &cl_ctx);
    SOA_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}

