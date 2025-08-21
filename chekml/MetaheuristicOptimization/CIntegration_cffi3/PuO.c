#include "PuO.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (stub, as in FirefA)
char *preprocess_kernel_source(const char *source);

void PuO_init_cl(PuOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->explore_kernel = clCreateKernel(cl_ctx->program, "exploration_phase", &err);
    check_cl_error(err, "clCreateKernel exploration_phase");
    cl_ctx->exploit_kernel = clCreateKernel(cl_ctx->program, "exploitation_phase", &err);
    check_cl_error(err, "clCreateKernel exploitation_phase");
}

void PuO_cleanup_cl(PuOCLContext *cl_ctx) {
    if (cl_ctx->exploit_kernel) clReleaseKernel(cl_ctx->exploit_kernel);
    if (cl_ctx->explore_kernel) clReleaseKernel(cl_ctx->explore_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void PuO_init_context(PuOContext *ctx, Optimizer *opt, PuOCLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;

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
    ctx->temp_position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer temp_position");
    ctx->y = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer y");
    ctx->z = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer z");
    ctx->beta2 = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer beta2");
    ctx->w = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer w");
    ctx->v = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer v");
    ctx->F1 = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer F1");
    ctx->F2 = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer F2");
    ctx->S1 = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer S1");
    ctx->S2 = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer S2");
    ctx->VEC = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer VEC");
    ctx->Xatack = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer Xatack");
    ctx->mbest = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer mbest");

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

void PuO_cleanup_context(PuOContext *ctx, PuOCLContext *cl_ctx) {
    if (ctx->mbest) clReleaseMemObject(ctx->mbest);
    if (ctx->Xatack) clReleaseMemObject(ctx->Xatack);
    if (ctx->VEC) clReleaseMemObject(ctx->VEC);
    if (ctx->S2) clReleaseMemObject(ctx->S2);
    if (ctx->S1) clReleaseMemObject(ctx->S1);
    if (ctx->F2) clReleaseMemObject(ctx->F2);
    if (ctx->F1) clReleaseMemObject(ctx->F1);
    if (ctx->v) clReleaseMemObject(ctx->v);
    if (ctx->w) clReleaseMemObject(ctx->w);
    if (ctx->beta2) clReleaseMemObject(ctx->beta2);
    if (ctx->z) clReleaseMemObject(ctx->z);
    if (ctx->y) clReleaseMemObject(ctx->y);
    if (ctx->temp_position) clReleaseMemObject(ctx->temp_position);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

void PuO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Puma Optimizer optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("PuO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open PuO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: PuO.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read PuO.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: PuO.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    PuOCLContext cl_ctx = {0};
    PuO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    PuOContext ctx = {0};
    PuO_init_context(&ctx, opt, &cl_ctx);

    // Initialize state
    PuOState state = {0};
    state.q_probability = PUO_Q_PROBABILITY;
    state.beta = PUO_BETA_FACTOR;
    state.PF[0] = PUO_PF1;
    state.PF[1] = PUO_PF2;
    state.PF[2] = PUO_PF3;
    state.mega_explore = PUO_MEGA_EXPLORE_INIT;
    state.mega_exploit = PUO_MEGA_EXPLOIT_INIT;
    state.unselected[0] = 1.0f;
    state.unselected[1] = 1.0f;
    for (int i = 0; i < 3; i++) {
        state.seq_time_explore[i] = 1.0f;
        state.seq_time_exploit[i] = 1.0f;
        state.seq_cost_explore[i] = 1.0f;
        state.seq_cost_exploit[i] = 1.0f;
    }
    state.flag_change = 1;

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
        double explore_time;
        double exploit_time;
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
    float initial_best_cost = best_fitness;

    positions = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer best_pos");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
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

    float costs_explore[3];
    float costs_exploit[3];

    // Unexperienced Phase
    for (int iter = 0; iter < 3; iter++) {
        // Exploration phase
        cl_event explore_event;
        float pCR = PUO_PCR_INITIAL;
        clSetKernelArg(cl_ctx.explore_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.explore_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.explore_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.explore_kernel, 3, sizeof(cl_mem), &ctx.temp_position);
        clSetKernelArg(cl_ctx.explore_kernel, 4, sizeof(cl_mem), &ctx.y);
        clSetKernelArg(cl_ctx.explore_kernel, 5, sizeof(cl_mem), &ctx.z);
        clSetKernelArg(cl_ctx.explore_kernel, 6, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.explore_kernel, 7, sizeof(cl_float), &pCR);
        clSetKernelArg(cl_ctx.explore_kernel, 8, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.explore_kernel, 9, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.explore_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &explore_event);
        check_cl_error(err, "clEnqueueNDRangeKernel exploration_phase");
        clFinish(cl_ctx.queue);

        double explore_time = 0.0;
        if (explore_event) {
            err = clGetEventProfilingInfo(explore_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(explore_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    explore_time = (end - start) / 1e6;
                    prof.explore_time += explore_time;
                }
            }
            clReleaseEvent(explore_event);
        }

        // Evaluate population
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
        costs_explore[iter] = new_best;
        if (new_best < ctx.best_fitness) {
            ctx.best_fitness = new_best;
            positions = (float *)malloc(dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer best_pos");
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
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

        // Exploitation phase
        cl_event exploit_event;
        float q_probability = state.q_probability;
        float beta = state.beta;
        // Compute mbest
        positions = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population for mbest");
        float *mbest = (float *)malloc(dim * sizeof(float));
        memset(mbest, 0, dim * sizeof(float));
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < dim; j++) {
                mbest[j] += positions[i * dim + j];
            }
        }
        for (int j = 0; j < dim; j++) {
            mbest[j] /= pop_size;
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.mbest, CL_TRUE, 0, dim * sizeof(cl_float), mbest, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer mbest");
        free(mbest);
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

        clSetKernelArg(cl_ctx.exploit_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.exploit_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.exploit_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.exploit_kernel, 3, sizeof(cl_mem), &ctx.best_position);
        clSetKernelArg(cl_ctx.exploit_kernel, 4, sizeof(cl_mem), &ctx.beta2);
        clSetKernelArg(cl_ctx.exploit_kernel, 5, sizeof(cl_mem), &ctx.w);
        clSetKernelArg(cl_ctx.exploit_kernel, 6, sizeof(cl_mem), &ctx.v);
        clSetKernelArg(cl_ctx.exploit_kernel, 7, sizeof(cl_mem), &ctx.F1);
        clSetKernelArg(cl_ctx.exploit_kernel, 8, sizeof(cl_mem), &ctx.F2);
        clSetKernelArg(cl_ctx.exploit_kernel, 9, sizeof(cl_mem), &ctx.S1);
        clSetKernelArg(cl_ctx.exploit_kernel, 10, sizeof(cl_mem), &ctx.S2);
        clSetKernelArg(cl_ctx.exploit_kernel, 11, sizeof(cl_mem), &ctx.VEC);
        clSetKernelArg(cl_ctx.exploit_kernel, 12, sizeof(cl_mem), &ctx.Xatack);
        clSetKernelArg(cl_ctx.exploit_kernel, 13, sizeof(cl_mem), &ctx.mbest);
        clSetKernelArg(cl_ctx.exploit_kernel, 14, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.exploit_kernel, 15, sizeof(cl_float), &q_probability);
        clSetKernelArg(cl_ctx.exploit_kernel, 16, sizeof(cl_float), &beta);
        clSetKernelArg(cl_ctx.exploit_kernel, 17, sizeof(cl_int), &iter);
        clSetKernelArg(cl_ctx.exploit_kernel, 18, sizeof(cl_int), &opt->max_iter);
        clSetKernelArg(cl_ctx.exploit_kernel, 19, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.exploit_kernel, 20, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.exploit_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &exploit_event);
        check_cl_error(err, "clEnqueueNDRangeKernel exploitation_phase");
        clFinish(cl_ctx.queue);

        double exploit_time = 0.0;
        if (exploit_event) {
            err = clGetEventProfilingInfo(exploit_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(exploit_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    exploit_time = (end - start) / 1e6;
                    prof.exploit_time += exploit_time;
                }
            }
            clReleaseEvent(exploit_event);
        }

        // Evaluate population
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
        new_best = fitness[0];
        best_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (fitness[i] < new_best) {
                new_best = fitness[i];
                best_idx = i;
            }
        }
        costs_exploit[iter] = new_best;
        if (new_best < ctx.best_fitness) {
            ctx.best_fitness = new_best;
            positions = (float *)malloc(dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer best_pos");
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
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

        state.seq_cost_explore[iter] = costs_explore[iter];
        state.seq_cost_exploit[iter] = costs_exploit[iter];
        state.seq_time_explore[iter] = explore_time;
        state.seq_time_exploit[iter] = exploit_time;
    }

    // Experienced Phase
    float sum_cost_explore = 0.0f, sum_cost_exploit = 0.0f;
    float sum_time_explore = 0.0f, sum_time_exploit = 0.0f;
    for (int i = 0; i < 3; i++) {
        sum_cost_explore += state.seq_cost_explore[i];
        sum_cost_exploit += state.seq_cost_exploit[i];
        sum_time_explore += state.seq_time_explore[i];
        sum_time_exploit += state.seq_time_exploit[i];
    }
    state.score_explore = sum_cost_explore / sum_time_explore;
    state.score_exploit = sum_cost_exploit / sum_time_exploit;

    // Initialize pf_f3
    state.pf_f3_size = 0;
    if (state.score_explore > state.score_exploit) {
        state.f3_explore = state.PF[2];
        if (state.pf_f3_size < PUO_PF_F3_MAX_SIZE) {
            state.pf_f3[state.pf_f3_size++] = state.f3_explore;
        }
    } else {
        state.f3_exploit = state.PF[2];
        if (state.pf_f3_size < PUO_PF_F3_MAX_SIZE) {
            state.pf_f3[state.pf_f3_size++] = state.f3_exploit;
        }
    }

    // Main optimization loop
    while (func_count < max_evals && iteration < opt->max_iter) {
        prof.count++;
        iteration++;

        // Update pCR
        float pCR = PUO_PCR_INITIAL * (1.0f - (float)iteration / opt->max_iter);

        // Select phase based on scores
        int use_exploration = state.score_explore > state.score_exploit || state.flag_change;

        if (use_exploration) {
            cl_event explore_event;
            clSetKernelArg(cl_ctx.explore_kernel, 0, sizeof(cl_mem), &ctx.population);
            clSetKernelArg(cl_ctx.explore_kernel, 1, sizeof(cl_mem), &ctx.fitness);
            clSetKernelArg(cl_ctx.explore_kernel, 2, sizeof(cl_mem), &ctx.bounds);
            clSetKernelArg(cl_ctx.explore_kernel, 3, sizeof(cl_mem), &ctx.temp_position);
            clSetKernelArg(cl_ctx.explore_kernel, 4, sizeof(cl_mem), &ctx.y);
            clSetKernelArg(cl_ctx.explore_kernel, 5, sizeof(cl_mem), &ctx.z);
            clSetKernelArg(cl_ctx.explore_kernel, 6, sizeof(cl_mem), &ctx.random_seeds);
            clSetKernelArg(cl_ctx.explore_kernel, 7, sizeof(cl_float), &pCR);
            clSetKernelArg(cl_ctx.explore_kernel, 8, sizeof(cl_int), &dim);
            clSetKernelArg(cl_ctx.explore_kernel, 9, sizeof(cl_int), &pop_size);

            err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.explore_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &explore_event);
            check_cl_error(err, "clEnqueueNDRangeKernel exploration_phase");
            clFinish(cl_ctx.queue);

            double explore_time = 0.0;
            if (explore_event) {
                err = clGetEventProfilingInfo(explore_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(explore_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        explore_time = (end - start) / 1e6;
                        prof.explore_time += explore_time;
                    }
                }
                clReleaseEvent(explore_event);
            }

            // Evaluate population
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
                err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
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

            // Update scores
            sum_cost_explore += new_best;
            sum_time_explore += explore_time;
            state.score_explore = sum_cost_explore / sum_time_explore;
        } else {
            cl_event exploit_event;
            float q_probability = state.q_probability;
            float beta = state.beta;
            // Compute mbest
            positions = (float *)malloc(pop_size * dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer population for mbest");
            float *mbest = (float *)malloc(dim * sizeof(float));
            memset(mbest, 0, dim * sizeof(float));
            for (int i = 0; i < pop_size; i++) {
                for (int j = 0; j < dim; j++) {
                    mbest[j] += positions[i * dim + j];
                }
            }
            for (int j = 0; j < dim; j++) {
                mbest[j] /= pop_size;
            }
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.mbest, CL_TRUE, 0, dim * sizeof(cl_float), mbest, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer mbest");
            free(mbest);
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

            clSetKernelArg(cl_ctx.exploit_kernel, 0, sizeof(cl_mem), &ctx.population);
            clSetKernelArg(cl_ctx.exploit_kernel, 1, sizeof(cl_mem), &ctx.fitness);
            clSetKernelArg(cl_ctx.exploit_kernel, 2, sizeof(cl_mem), &ctx.bounds);
            clSetKernelArg(cl_ctx.exploit_kernel, 3, sizeof(cl_mem), &ctx.best_position);
            clSetKernelArg(cl_ctx.exploit_kernel, 4, sizeof(cl_mem), &ctx.beta2);
            clSetKernelArg(cl_ctx.exploit_kernel, 5, sizeof(cl_mem), &ctx.w);
            clSetKernelArg(cl_ctx.exploit_kernel, 6, sizeof(cl_mem), &ctx.v);
            clSetKernelArg(cl_ctx.exploit_kernel, 7, sizeof(cl_mem), &ctx.F1);
            clSetKernelArg(cl_ctx.exploit_kernel, 8, sizeof(cl_mem), &ctx.F2);
            clSetKernelArg(cl_ctx.exploit_kernel, 9, sizeof(cl_mem), &ctx.S1);
            clSetKernelArg(cl_ctx.exploit_kernel, 10, sizeof(cl_mem), &ctx.S2);
            clSetKernelArg(cl_ctx.exploit_kernel, 11, sizeof(cl_mem), &ctx.VEC);
            clSetKernelArg(cl_ctx.exploit_kernel, 12, sizeof(cl_mem), &ctx.Xatack);
            clSetKernelArg(cl_ctx.exploit_kernel, 13, sizeof(cl_mem), &ctx.mbest);
            clSetKernelArg(cl_ctx.exploit_kernel, 14, sizeof(cl_mem), &ctx.random_seeds);
            clSetKernelArg(cl_ctx.exploit_kernel, 15, sizeof(cl_float), &q_probability);
            clSetKernelArg(cl_ctx.exploit_kernel, 16, sizeof(cl_float), &beta);
            clSetKernelArg(cl_ctx.exploit_kernel, 17, sizeof(cl_int), &iteration);
            clSetKernelArg(cl_ctx.exploit_kernel, 18, sizeof(cl_int), &opt->max_iter);
            clSetKernelArg(cl_ctx.exploit_kernel, 19, sizeof(cl_int), &dim);
            clSetKernelArg(cl_ctx.exploit_kernel, 20, sizeof(cl_int), &pop_size);

            err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.exploit_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &exploit_event);
            check_cl_error(err, "clEnqueueNDRangeKernel exploitation_phase");
            clFinish(cl_ctx.queue);

            double exploit_time = 0.0;
            if (exploit_event) {
                err = clGetEventProfilingInfo(exploit_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(exploit_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        exploit_time = (end - start) / 1e6;
                        prof.exploit_time += exploit_time;
                    }
                }
                clReleaseEvent(exploit_event);
            }

            // Evaluate population
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
                err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
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

            // Update scores
            sum_cost_exploit += new_best;
            sum_time_exploit += exploit_time;
            state.score_exploit = sum_cost_exploit / sum_time_exploit;
        }

        // Update pf_f3
        if (state.score_explore > state.score_exploit) {
            state.f3_explore = state.PF[2] * (1.0f - (float)iteration / opt->max_iter);
            if (state.pf_f3_size < PUO_PF_F3_MAX_SIZE) {
                state.pf_f3[state.pf_f3_size++] = state.f3_explore;
            }
        } else {
            state.f3_exploit = state.PF[2] * (1.0f - (float)iteration / opt->max_iter);
            if (state.pf_f3_size < PUO_PF_F3_MAX_SIZE) {
                state.pf_f3[state.pf_f3_size++] = state.f3_exploit;
            }
        }

        // Update flag_change
        state.flag_change = (iteration % 10 == 0) ? !state.flag_change : state.flag_change;
    }

    // Store final best solution
    positions = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer final best_pos");
    for (int i = 0; i < dim; i++) {
        opt->best_solution.position[i] = positions[i];
    }
    opt->best_solution.fitness = ctx.best_fitness;
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

    free(fitness);
    clReleaseMemObject(best_fitness_buf);

    // Print profiling results
    printf("Initial Population: %.3f ms\n", prof.init_pop_time);
    printf("Exploration Phase: %.3f ms\n", prof.explore_time);
    printf("Exploitation Phase: %.3f ms\n", prof.exploit_time);
    printf("Total Update (Explore + Exploit): %.3f ms\n", prof.explore_time + prof.exploit_time);
    printf("Memory Read: %.3f ms\n", prof.read_time);
    printf("Memory Write: %.3f ms\n", prof.write_time);
    if (prof.count > 0) {
        printf("\nAverage per iteration (%d iterations):\n", prof.count);
        printf("Exploration: %.3f ms\n", prof.explore_time / prof.count);
        printf("Exploitation: %.3f ms\n", prof.exploit_time / prof.count);
        printf("Update (Explore + Exploit): %.3f ms\n", (prof.explore_time + prof.exploit_time) / prof.count);
        printf("Read: %.3f ms\n", prof.read_time / prof.count);
        printf("Write: %.3f ms\n", prof.write_time / prof.count);
    }

    // Cleanup
    PuO_cleanup_context(&ctx, &cl_ctx);
    PuO_cleanup_cl(&cl_ctx);

    printf("Optimization complete. Best fitness: %.6f\n", ctx.best_fitness);
}
