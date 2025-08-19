#include "CheRO.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (placeholder, as in FirefA)
char *preprocess_kernel_source(const char *source);

void CheRO_init_cl(CheROCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->on_wall_kernel = clCreateKernel(cl_ctx->program, "on_wall_collision", &err);
    check_cl_error(err, "clCreateKernel on_wall_collision");
    cl_ctx->decomp_kernel = clCreateKernel(cl_ctx->program, "decomposition", &err);
    check_cl_error(err, "clCreateKernel decomposition");
    cl_ctx->inter_coll_kernel = clCreateKernel(cl_ctx->program, "inter_molecular_collision", &err);
    check_cl_error(err, "clCreateKernel inter_molecular_collision");
    cl_ctx->synthesis_kernel = clCreateKernel(cl_ctx->program, "synthesis", &err);
    check_cl_error(err, "clCreateKernel synthesis");
    cl_ctx->elim_kernel = clCreateKernel(cl_ctx->program, "elimination_phase", &err);
    check_cl_error(err, "clCreateKernel elimination_phase");
}

void CheRO_cleanup_cl(CheROCLContext *cl_ctx) {
    if (cl_ctx->elim_kernel) clReleaseKernel(cl_ctx->elim_kernel);
    if (cl_ctx->synthesis_kernel) clReleaseKernel(cl_ctx->synthesis_kernel);
    if (cl_ctx->inter_coll_kernel) clReleaseKernel(cl_ctx->inter_coll_kernel);
    if (cl_ctx->decomp_kernel) clReleaseKernel(cl_ctx->decomp_kernel);
    if (cl_ctx->on_wall_kernel) clReleaseKernel(cl_ctx->on_wall_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void CheRO_init_context(CheROContext *ctx, Optimizer *opt, CheROCLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;

    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer position");
    ctx->pe = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer pe");
    ctx->ke = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer ke");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
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
}

void CheRO_cleanup_context(CheROContext *ctx, CheROCLContext *cl_ctx) {
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->ke) clReleaseMemObject(ctx->ke);
    if (ctx->pe) clReleaseMemObject(ctx->pe);
    if (ctx->position) clReleaseMemObject(ctx->position);
}

void CheRO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Chemical Reaction Optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device ||
        !opt->population_buffer || !opt->fitness_buffer ||
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("CheRO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open CheRO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: CheRO.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read CheRO.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: CheRO.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    CheROCLContext cl_ctx = {0};
    CheRO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    CheROContext ctx = {0};
    CheRO_init_context(&ctx, opt, &cl_ctx);

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
        double on_wall_time;
        double decomp_time;
        double inter_coll_time;
        double synthesis_time;
        double elim_time;
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
    clSetKernelArg(cl_ctx.init_pop_kernel, 0, sizeof(cl_mem), &ctx.position);
    clSetKernelArg(cl_ctx.init_pop_kernel, 1, sizeof(cl_mem), &ctx.pe);
    clSetKernelArg(cl_ctx.init_pop_kernel, 2, sizeof(cl_mem), &ctx.ke);
    clSetKernelArg(cl_ctx.init_pop_kernel, 3, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 4, sizeof(cl_mem), &ctx.best_position);
    clSetKernelArg(cl_ctx.init_pop_kernel, 5, sizeof(cl_mem), &best_fitness_buf);
    clSetKernelArg(cl_ctx.init_pop_kernel, 6, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 7, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_pop_kernel, 8, sizeof(cl_int), &pop_size);

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
                prof.init_pop_time += (end - start) / 1e6;
            }
        }
        clReleaseEvent(init_event);
    }

    // Evaluate initial population on CPU
    cl_event read_event, write_event;
    float *positions = (float *)malloc(pop_size * dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.position, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer position");
    float *pe = (float *)malloc(pop_size * sizeof(float));
    for (int i = 0; i < pop_size; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
        pe[i] = (float)objective_function(pos);
        free(pos);
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.pe, CL_TRUE, 0, pop_size * sizeof(cl_float), pe, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer pe");
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
    float best_fitness = pe[0];
    int best_idx = 0;
    for (int i = 1; i < pop_size; i++) {
        if (pe[i] < best_fitness) {
            best_fitness = pe[i];
            best_idx = i;
        }
    }
    ctx.best_fitness = best_fitness;
    free(pe);

    positions = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.position, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
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

    // Main optimization loop
    while (func_count < max_evals) {
        iteration++;
        printf("Iteration %d\n", iteration);

        // On-wall collision
        cl_event on_wall_event;
        cl_float mole_coll = CheRO_MOLE_COLL;
        cl_float alpha = CheRO_ALPHA;
        clSetKernelArg(cl_ctx.on_wall_kernel, 0, sizeof(cl_mem), &ctx.position);
        clSetKernelArg(cl_ctx.on_wall_kernel, 1, sizeof(cl_mem), &ctx.pe);
        clSetKernelArg(cl_ctx.on_wall_kernel, 2, sizeof(cl_mem), &ctx.ke);
        clSetKernelArg(cl_ctx.on_wall_kernel, 3, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.on_wall_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.on_wall_kernel, 5, sizeof(cl_float), &mole_coll);
        clSetKernelArg(cl_ctx.on_wall_kernel, 6, sizeof(cl_float), &alpha);
        clSetKernelArg(cl_ctx.on_wall_kernel, 7, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.on_wall_kernel, 8, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.on_wall_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &on_wall_event);
        check_cl_error(err, "clEnqueueNDRangeKernel on_wall_collision");
        clFinish(cl_ctx.queue);

        double on_wall_time = 0.0;
        if (on_wall_event) {
            err = clGetEventProfilingInfo(on_wall_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(on_wall_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    on_wall_time = (end - start) / 1e6;
                    prof.on_wall_time += on_wall_time;
                }
            }
            clReleaseEvent(on_wall_event);
        }

        // Decomposition
        cl_event decomp_event;
        cl_float split_ratio = CheRO_SPLIT_RATIO;
        clSetKernelArg(cl_ctx.decomp_kernel, 0, sizeof(cl_mem), &ctx.position);
        clSetKernelArg(cl_ctx.decomp_kernel, 1, sizeof(cl_mem), &ctx.pe);
        clSetKernelArg(cl_ctx.decomp_kernel, 2, sizeof(cl_mem), &ctx.ke);
        clSetKernelArg(cl_ctx.decomp_kernel, 3, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.decomp_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.decomp_kernel, 5, sizeof(cl_float), &mole_coll);
        clSetKernelArg(cl_ctx.decomp_kernel, 6, sizeof(cl_float), &alpha);
        clSetKernelArg(cl_ctx.decomp_kernel, 7, sizeof(cl_float), &split_ratio);
        clSetKernelArg(cl_ctx.decomp_kernel, 8, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.decomp_kernel, 9, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.decomp_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &decomp_event);
        check_cl_error(err, "clEnqueueNDRangeKernel decomposition");
        clFinish(cl_ctx.queue);

        double decomp_time = 0.0;
        if (decomp_event) {
            err = clGetEventProfilingInfo(decomp_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(decomp_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    decomp_time = (end - start) / 1e6;
                    prof.decomp_time += decomp_time;
                }
            }
            clReleaseEvent(decomp_event);
        }

        // Inter-molecular collision
        cl_event inter_coll_event;
        clSetKernelArg(cl_ctx.inter_coll_kernel, 0, sizeof(cl_mem), &ctx.position);
        clSetKernelArg(cl_ctx.inter_coll_kernel, 1, sizeof(cl_mem), &ctx.pe);
        clSetKernelArg(cl_ctx.inter_coll_kernel, 2, sizeof(cl_mem), &ctx.ke);
        clSetKernelArg(cl_ctx.inter_coll_kernel, 3, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.inter_coll_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.inter_coll_kernel, 5, sizeof(cl_float), &mole_coll);
        clSetKernelArg(cl_ctx.inter_coll_kernel, 6, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.inter_coll_kernel, 7, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.inter_coll_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &inter_coll_event);
        check_cl_error(err, "clEnqueueNDRangeKernel inter_molecular_collision");
        clFinish(cl_ctx.queue);

        double inter_coll_time = 0.0;
        if (inter_coll_event) {
            err = clGetEventProfilingInfo(inter_coll_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(inter_coll_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    inter_coll_time = (end - start) / 1e6;
                    prof.inter_coll_time += inter_coll_time;
                }
            }
            clReleaseEvent(inter_coll_event);
        }

        // Synthesis
        cl_event synthesis_event;
        cl_float beta = CheRO_BETA;
        clSetKernelArg(cl_ctx.synthesis_kernel, 0, sizeof(cl_mem), &ctx.position);
        clSetKernelArg(cl_ctx.synthesis_kernel, 1, sizeof(cl_mem), &ctx.pe);
        clSetKernelArg(cl_ctx.synthesis_kernel, 2, sizeof(cl_mem), &ctx.ke);
        clSetKernelArg(cl_ctx.synthesis_kernel, 3, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.synthesis_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.synthesis_kernel, 5, sizeof(cl_float), &mole_coll);
        clSetKernelArg(cl_ctx.synthesis_kernel, 6, sizeof(cl_float), &beta);
        clSetKernelArg(cl_ctx.synthesis_kernel, 7, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.synthesis_kernel, 8, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.synthesis_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &synthesis_event);
        check_cl_error(err, "clEnqueueNDRangeKernel synthesis");
        clFinish(cl_ctx.queue);

        double synthesis_time = 0.0;
        if (synthesis_event) {
            err = clGetEventProfilingInfo(synthesis_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(synthesis_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    synthesis_time = (end - start) / 1e6;
                    prof.synthesis_time += synthesis_time;
                }
            }
            clReleaseEvent(synthesis_event);
        }

        // Evaluate population on CPU
        positions = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.position, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer position");
        pe = (float *)malloc(pop_size * sizeof(float));
        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            pe[i] = (float)objective_function(pos);
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.pe, CL_TRUE, 0, pop_size * sizeof(cl_float), pe, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer pe");
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
        float new_best = pe[0];
        best_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (pe[i] < new_best) {
                new_best = pe[i];
                best_idx = i;
            }
        }
        if (new_best < ctx.best_fitness) {
            ctx.best_fitness = new_best;
            positions = (float *)malloc(dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.position, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
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
        free(pe);

        // Elimination phase
        cl_event elim_event;
        cl_float elim_ratio = CheRO_ELIMINATION_RATIO;
        cl_float initial_ke = CheRO_INITIAL_KE;
        clSetKernelArg(cl_ctx.elim_kernel, 0, sizeof(cl_mem), &ctx.position);
        clSetKernelArg(cl_ctx.elim_kernel, 1, sizeof(cl_mem), &ctx.pe);
        clSetKernelArg(cl_ctx.elim_kernel, 2, sizeof(cl_mem), &ctx.ke);
        clSetKernelArg(cl_ctx.elim_kernel, 3, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.elim_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.elim_kernel, 5, sizeof(cl_float), &elim_ratio);
        clSetKernelArg(cl_ctx.elim_kernel, 6, sizeof(cl_float), &initial_ke);
        clSetKernelArg(cl_ctx.elim_kernel, 7, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.elim_kernel, 8, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.elim_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &elim_event);
        check_cl_error(err, "clEnqueueNDRangeKernel elimination_phase");
        clFinish(cl_ctx.queue);

        double elim_time = 0.0;
        if (elim_event) {
            err = clGetEventProfilingInfo(elim_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(elim_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    elim_time = (end - start) / 1e6;
                    prof.elim_time += elim_time;
                }
            }
            clReleaseEvent(elim_event);
        }

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.best_fitness);
        printf("Profiling (ms): On-Wall = %.3f, Decomposition = %.3f, Inter-Collision = %.3f, Synthesis = %.3f, Elimination = %.3f\n",
               on_wall_time, decomp_time, inter_coll_time, synthesis_time, elim_time);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_pop_time);
    printf("On-Wall Collision: %.3f\n", prof.on_wall_time);
    printf("Decomposition: %.3f\n", prof.decomp_time);
    printf("Inter-Molecular Collision: %.3f\n", prof.inter_coll_time);
    printf("Synthesis: %.3f\n", prof.synthesis_time);
    printf("Elimination: %.3f\n", prof.elim_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("On-Wall Collision: %.3f\n", prof.on_wall_time / prof.count);
        printf("Decomposition: %.3f\n", prof.decomp_time / prof.count);
        printf("Inter-Molecular Collision: %.3f\n", prof.inter_coll_time / prof.count);
        printf("Synthesis: %.3f\n", prof.synthesis_time / prof.count);
        printf("Elimination: %.3f\n", prof.elim_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Copy best solution to optimizer
    float *best_pos = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer final best_pos");
    opt->best_solution.fitness = ctx.best_fitness;
    for (int i = 0; i < dim; i++) opt->best_solution.position[i] = best_pos[i];
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
    CheRO_cleanup_context(&ctx, &cl_ctx);
    CheRO_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
