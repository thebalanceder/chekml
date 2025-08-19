#include "KHO.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (stub, as no specific preprocessing is needed)
char *preprocess_kernel_source(const char *source);

void KHO_init_cl(KHOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->motion_kernel = clCreateKernel(cl_ctx->program, "movement_induced_phase", &err);
    check_cl_error(err, "clCreateKernel movement_induced");
    cl_ctx->foraging_kernel = clCreateKernel(cl_ctx->program, "foraging_motion_phase", &err);
    check_cl_error(err, "clCreateKernel foraging_motion");
    cl_ctx->diffusion_kernel = clCreateKernel(cl_ctx->program, "physical_diffusion_phase", &err);
    check_cl_error(err, "clCreateKernel physical_diffusion");
    cl_ctx->crossover_kernel = clCreateKernel(cl_ctx->program, "crossover_phase", &err);
    check_cl_error(err, "clCreateKernel crossover");
    cl_ctx->update_pos_kernel = clCreateKernel(cl_ctx->program, "update_positions", &err);
    check_cl_error(err, "clCreateKernel update_positions");
}

void KHO_cleanup_cl(KHOCLContext *cl_ctx) {
    if (cl_ctx->update_pos_kernel) clReleaseKernel(cl_ctx->update_pos_kernel);
    if (cl_ctx->crossover_kernel) clReleaseKernel(cl_ctx->crossover_kernel);
    if (cl_ctx->diffusion_kernel) clReleaseKernel(cl_ctx->diffusion_kernel);
    if (cl_ctx->foraging_kernel) clReleaseKernel(cl_ctx->foraging_kernel);
    if (cl_ctx->motion_kernel) clReleaseKernel(cl_ctx->motion_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void KHO_init_context(KHOContext *ctx, Optimizer *opt, KHOCLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;
    ctx->Kf = INFINITY;

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
    ctx->N = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer N");
    ctx->F = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer F");
    ctx->D = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer D");
    ctx->local_best_pos = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer local_best_pos");
    ctx->local_best_fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer local_best_fitness");
    ctx->Xf = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer Xf");
    ctx->K = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer K");
    ctx->Kib = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer Kib");

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

    // Initialize motion vectors to zero
    float *zeros = (float *)calloc(pop_size * dim, sizeof(float));
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->N, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), zeros, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer N");
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->F, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), zeros, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer F");
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->D, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), zeros, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer D");
    free(zeros);
}

void KHO_cleanup_context(KHOContext *ctx, KHOCLContext *cl_ctx) {
    if (ctx->Kib) clReleaseMemObject(ctx->Kib);
    if (ctx->K) clReleaseMemObject(ctx->K);
    if (ctx->Xf) clReleaseMemObject(ctx->Xf);
    if (ctx->local_best_fitness) clReleaseMemObject(ctx->local_best_fitness);
    if (ctx->local_best_pos) clReleaseMemObject(ctx->local_best_pos);
    if (ctx->D) clReleaseMemObject(ctx->D);
    if (ctx->F) clReleaseMemObject(ctx->F);
    if (ctx->N) clReleaseMemObject(ctx->N);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

void KHO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Krill Herd Optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device ||
        !opt->population_buffer || !opt->fitness_buffer ||
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("KHO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open KHO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: KHO.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read KHO.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: KHO.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    KHOCLContext cl_ctx = {0};
    KHO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    KHOContext ctx = {0};
    KHO_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * pop_size : MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Compute Dt
    float Dt = 0.0f;
    for (int j = 0; j < dim; j++) {
        Dt += fabsf((float)opt->bounds[2 * j + 1] - (float)opt->bounds[2 * j]);
    }
    Dt /= (2.0f * dim);

    // Profiling data
    typedef struct {
        double init_pop_time;
        double motion_time;
        double foraging_time;
        double diffusion_time;
        double crossover_time;
        double update_pos_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;
    cl_mem best_fitness_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_fitness");
    cl_mem Kf_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer Kf");

    // Initialize population
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_pop_kernel, 0, sizeof(cl_mem), &ctx.population);
    clSetKernelArg(cl_ctx.init_pop_kernel, 1, sizeof(cl_mem), &ctx.fitness);
    clSetKernelArg(cl_ctx.init_pop_kernel, 2, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 3, sizeof(cl_mem), &ctx.best_position);
    clSetKernelArg(cl_ctx.init_pop_kernel, 4, sizeof(cl_mem), &best_fitness_buf);
    clSetKernelArg(cl_ctx.init_pop_kernel, 5, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 6, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_pop_kernel, 7, sizeof(cl_int), &pop_size);

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
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.K, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer K");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.Kib, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer Kib");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.local_best_fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer local_best_fitness");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.local_best_pos, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer local_best_pos");

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
    for (int j = 0; j < dim; j++) {
        best_pos[j] = positions[best_idx * dim + j];
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer best_pos");
    err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &best_fitness, 0, NULL, NULL);
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

        // Compute food position (Xf) on CPU
        float *K = (float *)malloc(pop_size * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.K, CL_TRUE, 0, pop_size * sizeof(cl_float), K, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer K");
        float sum_inv_K = 0.0f;
        for (int i = 0; i < pop_size; i++) {
            sum_inv_K += 1.0f / (K[i] + 1e-10f);
        }
        float *Sf = (float *)calloc(dim, sizeof(float));
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < dim; j++) {
                Sf[j] += positions[i * dim + j] / (K[i] + 1e-10f);
            }
        }
        float *Xf = (float *)malloc(dim * sizeof(float));
        for (int j = 0; j < dim; j++) {
            Xf[j] = Sf[j] / sum_inv_K;
            float lb = (float)opt->bounds[2 * j];
            float ub = (float)opt->bounds[2 * j + 1];
            if (Xf[j] < lb) Xf[j] = lb + (1.0f - (lb / (lb + best_pos[j]))) * best_pos[j];
            else if (Xf[j] > ub) Xf[j] = ub + (1.0f - (ub / (ub + best_pos[j]))) * best_pos[j];
        }
        free(Sf);
        double *Xf_double = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) Xf_double[j] = Xf[j];
        ctx.Kf = (float)objective_function(Xf_double);
        free(Xf_double);
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.Xf, CL_TRUE, 0, dim * sizeof(cl_float), Xf, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer Xf");
        err = clEnqueueWriteBuffer(cl_ctx.queue, Kf_buf, CL_TRUE, 0, sizeof(cl_float), &ctx.Kf, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer Kf");
        free(Xf);

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

        // Compute inertia weight and Kw_Kgb
        float w = KHO_INERTIA_MIN + KHO_INERTIA_MAX * (1.0f - (float)iteration / opt->max_iter);
        float max_K = K[0];
        for (int i = 1; i < pop_size; i++) {
            if (K[i] > max_K) max_K = K[i];
        }
        float Kw_Kgb = max_K - ctx.best_fitness;
        free(K);

        // Movement-induced phase
        cl_event motion_event;
        clSetKernelArg(cl_ctx.motion_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.motion_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.motion_kernel, 2, sizeof(cl_mem), &ctx.best_position);
        clSetKernelArg(cl_ctx.motion_kernel, 3, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.motion_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.motion_kernel, 5, sizeof(cl_mem), &ctx.N);
        clSetKernelArg(cl_ctx.motion_kernel, 6, sizeof(cl_float), &w);
        clSetKernelArg(cl_ctx.motion_kernel, 7, sizeof(cl_float), &Kw_Kgb);
        clSetKernelArg(cl_ctx.motion_kernel, 8, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.motion_kernel, 9, sizeof(cl_int), &pop_size);
        clSetKernelArg(cl_ctx.motion_kernel, 10, sizeof(cl_float), &ctx.best_fitness);
        clSetKernelArg(cl_ctx.motion_kernel, 11, sizeof(cl_int), &iteration);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.motion_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &motion_event);
        check_cl_error(err, "clEnqueueNDRangeKernel movement_induced");
        clFinish(cl_ctx.queue);

        if (motion_event) {
            err = clGetEventProfilingInfo(motion_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(motion_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.motion_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(motion_event);
        }

        // Foraging motion phase
        cl_event foraging_event;
        clSetKernelArg(cl_ctx.foraging_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.foraging_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.foraging_kernel, 2, sizeof(cl_mem), &ctx.Xf);
        clSetKernelArg(cl_ctx.foraging_kernel, 3, sizeof(cl_mem), &ctx.local_best_pos);
        clSetKernelArg(cl_ctx.foraging_kernel, 4, sizeof(cl_mem), &ctx.local_best_fitness);
        clSetKernelArg(cl_ctx.foraging_kernel, 5, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.foraging_kernel, 6, sizeof(cl_mem), &ctx.F);
        clSetKernelArg(cl_ctx.foraging_kernel, 7, sizeof(cl_float), &w);
        clSetKernelArg(cl_ctx.foraging_kernel, 8, sizeof(cl_float), &Kw_Kgb);
        clSetKernelArg(cl_ctx.foraging_kernel, 9, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.foraging_kernel, 10, sizeof(cl_int), &pop_size);
        clSetKernelArg(cl_ctx.foraging_kernel, 11, sizeof(cl_float), &ctx.Kf);
        clSetKernelArg(cl_ctx.foraging_kernel, 12, sizeof(cl_int), &iteration);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.foraging_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &foraging_event);
        check_cl_error(err, "clEnqueueNDRangeKernel foraging_motion");
        clFinish(cl_ctx.queue);

        if (foraging_event) {
            err = clGetEventProfilingInfo(foraging_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(foraging_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.foraging_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(foraging_event);
        }

        // Physical diffusion phase
        cl_event diffusion_event;
        clSetKernelArg(cl_ctx.diffusion_kernel, 0, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.diffusion_kernel, 1, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.diffusion_kernel, 2, sizeof(cl_mem), &ctx.D);
        clSetKernelArg(cl_ctx.diffusion_kernel, 3, sizeof(cl_int), &iteration);
        clSetKernelArg(cl_ctx.diffusion_kernel, 4, sizeof(cl_float), &Kw_Kgb);
        clSetKernelArg(cl_ctx.diffusion_kernel, 5, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.diffusion_kernel, 6, sizeof(cl_int), &pop_size);
        clSetKernelArg(cl_ctx.diffusion_kernel, 7, sizeof(cl_float), &ctx.best_fitness);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.diffusion_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &diffusion_event);
        check_cl_error(err, "clEnqueueNDRangeKernel physical_diffusion");
        clFinish(cl_ctx.queue);

        if (diffusion_event) {
            err = clGetEventProfilingInfo(diffusion_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(diffusion_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.diffusion_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(diffusion_event);
        }

        // Crossover phase
        cl_event crossover_event;
        clSetKernelArg(cl_ctx.crossover_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.crossover_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.crossover_kernel, 2, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.crossover_kernel, 3, sizeof(cl_float), &Kw_Kgb);
        clSetKernelArg(cl_ctx.crossover_kernel, 4, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.crossover_kernel, 5, sizeof(cl_int), &pop_size);
        clSetKernelArg(cl_ctx.crossover_kernel, 6, sizeof(cl_float), &ctx.best_fitness);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.crossover_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &crossover_event);
        check_cl_error(err, "clEnqueueNDRangeKernel crossover");
        clFinish(cl_ctx.queue);

        if (crossover_event) {
            err = clGetEventProfilingInfo(crossover_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(crossover_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.crossover_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(crossover_event);
        }

        // Update positions
        cl_event update_pos_event;
        clSetKernelArg(cl_ctx.update_pos_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.update_pos_kernel, 1, sizeof(cl_mem), &ctx.N);
        clSetKernelArg(cl_ctx.update_pos_kernel, 2, sizeof(cl_mem), &ctx.F);
        clSetKernelArg(cl_ctx.update_pos_kernel, 3, sizeof(cl_mem), &ctx.D);
        clSetKernelArg(cl_ctx.update_pos_kernel, 4, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.update_pos_kernel, 5, sizeof(cl_mem), &ctx.best_position);
        clSetKernelArg(cl_ctx.update_pos_kernel, 6, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.update_pos_kernel, 7, sizeof(cl_float), &Dt);
        clSetKernelArg(cl_ctx.update_pos_kernel, 8, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.update_pos_kernel, 9, sizeof(cl_int), &pop_size);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_pos_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &update_pos_event);
        check_cl_error(err, "clEnqueueNDRangeKernel update_positions");
        clFinish(cl_ctx.queue);

        if (update_pos_event) {
            err = clGetEventProfilingInfo(update_pos_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(update_pos_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.update_pos_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(update_pos_event);
        }

        // Evaluate population on CPU
        free(positions);
        positions = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population");
        free(fitness);
        fitness = (float *)malloc(pop_size * sizeof(float));
        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            fitness[i] = (float)objective_function(pos);
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer fitness");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.K, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer K");
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

        // Update local and global best solutions
        float *local_best_fitness = (float *)malloc(pop_size * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.local_best_fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), local_best_fitness, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer local_best_fitness");
        float *local_best_pos = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.local_best_pos, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), local_best_pos, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer local_best_pos");

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

        for (int i = 0; i < pop_size; i++) {
            if (fitness[i] < local_best_fitness[i]) {
                local_best_fitness[i] = fitness[i];
                for (int j = 0; j < dim; j++) {
                    local_best_pos[i * dim + j] = positions[i * dim + j];
                }
            }
        }

        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.local_best_fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), local_best_fitness, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer local_best_fitness");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.local_best_pos, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), local_best_pos, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer local_best_pos");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.Kib, CL_TRUE, 0, pop_size * sizeof(cl_float), local_best_fitness, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer Kib");

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

        float new_best = fitness[0];
        int best_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (fitness[i] < new_best) {
                new_best = fitness[i];
                best_idx = i;
            }
        }
        if (new_best < ctx.best_fitness) {
            ctx.best_fitness = new_best;
            free(best_pos);
            best_pos = (float *)malloc(dim * sizeof(float));
            for (int j = 0; j < dim; j++) {
                best_pos[j] = positions[best_idx * dim + j];
            }
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer best_pos");
            err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &new_best, 0, NULL, NULL);
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

        free(local_best_fitness);
        free(local_best_pos);

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.best_fitness);

        // Print profiling for this iteration
        printf("Profiling (ms): Motion = %.3f, Foraging = %.3f, Diffusion = %.3f, Crossover = %.3f, Update Pos = %.3f\n",
               prof.motion_time / prof.count, prof.foraging_time / prof.count, prof.diffusion_time / prof.count,
               prof.crossover_time / prof.count, prof.update_pos_time / prof.count);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_pop_time);
    printf("Movement-Induced Phase: %.3f\n", prof.motion_time);
    printf("Foraging Motion Phase: %.3f\n", prof.foraging_time);
    printf("Physical Diffusion Phase: %.3f\n", prof.diffusion_time);
    printf("Crossover Phase: %.3f\n", prof.crossover_time);
    printf("Update Positions: %.3f\n", prof.update_pos_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Movement-Induced Phase: %.3f\n", prof.motion_time / prof.count);
        printf("Foraging Motion Phase: %.3f\n", prof.foraging_time / prof.count);
        printf("Physical Diffusion Phase: %.3f\n", prof.diffusion_time / prof.count);
        printf("Crossover Phase: %.3f\n", prof.crossover_time / prof.count);
        printf("Update Positions: %.3f\n", prof.update_pos_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Copy best solution to optimizer
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer final best_pos");
    opt->best_solution.fitness = ctx.best_fitness;
    for (int i = 0; i < dim; i++) opt->best_solution.position[i] = best_pos[i];

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
    free(positions);
    free(fitness);
    free(best_pos);
    clReleaseMemObject(best_fitness_buf);
    clReleaseMemObject(Kf_buf);
    KHO_cleanup_context(&ctx, &cl_ctx);
    KHO_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
