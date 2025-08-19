#include "HHO.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>

// Define MAX_EVALS_DEFAULT if not provided
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 10000
#endif

void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source to fix address space for rand_float
char *preprocess_kernel_source(const char *source);

void HHO_init_cl(HHOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
    cl_int err;

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

    // Preprocess kernel source
    char *processed_source = preprocess_kernel_source(kernel_source);

    cl_ctx->context = opt->context;
    cl_ctx->device = opt->device;

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
        } else {
            fprintf(stderr, "Error building program: %d\nFailed to allocate memory for build log\n", err);
        }
        clReleaseProgram(cl_ctx->program);
        free(processed_source);
        exit(1);
    }

    free(processed_source);

    cl_ctx->init_pop_kernel = clCreateKernel(cl_ctx->program, "initialize_population", &err);
    check_cl_error(err, "clCreateKernel init_pop");
    cl_ctx->exploration_kernel = clCreateKernel(cl_ctx->program, "exploration_phase", &err);
    check_cl_error(err, "clCreateKernel exploration");
    cl_ctx->exploitation_kernel = clCreateKernel(cl_ctx->program, "exploitation_phase", &err);
    check_cl_error(err, "clCreateKernel exploitation");
    cl_ctx->levy_flight_kernel = clCreateKernel(cl_ctx->program, "levy_flight", &err);
    check_cl_error(err, "clCreateKernel levy_flight");
}

void HHO_cleanup_cl(HHOCLContext *cl_ctx) {
    if (cl_ctx->levy_flight_kernel) clReleaseKernel(cl_ctx->levy_flight_kernel);
    if (cl_ctx->exploitation_kernel) clReleaseKernel(cl_ctx->exploitation_kernel);
    if (cl_ctx->exploration_kernel) clReleaseKernel(cl_ctx->exploration_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void HHO_init_context(HHOContext *ctx, Optimizer *opt, HHOCLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;

    int pop_size = opt->population_size;
    int dim = opt->dim;

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

    cl_uint *seeds = (cl_uint *)malloc(pop_size * sizeof(cl_uint));
    srand(time(NULL));
    for (int i = 0; i < pop_size; i++) seeds[i] = rand();
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->random_seeds, CL_TRUE, 0, pop_size * sizeof(cl_uint), seeds, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer seeds");
    free(seeds);

    float *bounds_float = (float *)malloc(dim * 2 * sizeof(float));
    for (int i = 0; i < dim * 2; i++) bounds_float[i] = (float)opt->bounds[i];
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->bounds, CL_TRUE, 0, dim * 2 * sizeof(cl_float), bounds_float, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer bounds");
    free(bounds_float);
}

void HHO_cleanup_context(HHOContext *ctx, HHOCLContext *cl_ctx) {
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

void HHO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU optimization with HHO...\n");

    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    FILE *fp = fopen("HHO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open HHO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: HHO.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read HHO.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: HHO.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    HHOCLContext cl_ctx = {0};
    HHO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    HHOContext ctx = {0};
    HHO_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;

    int max_evals = opt->max_iter > 0 ? opt->max_iter * pop_size : MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_pop_time;
        double exploration_time;
        double exploitation_time;
        double levy_flight_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;
    cl_mem best_fitness_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_fitness");
    cl_mem mean_pos_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer mean_pos");
    cl_mem levy_step_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer levy_step");

    // Initialize population
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_pop_kernel, 0, sizeof(cl_mem), &ctx.population);
    clSetKernelArg(cl_ctx.init_pop_kernel, 1, sizeof(cl_mem), &ctx.fitness);
    clSetKernelArg(cl_ctx.init_pop_kernel, 2, sizeof(cl_mem), &ctx.best_position);
    clSetKernelArg(cl_ctx.init_pop_kernel, 3, sizeof(cl_mem), &best_fitness_buf);
    clSetKernelArg(cl_ctx.init_pop_kernel, 4, sizeof(cl_mem), &ctx.bounds);
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
                prof.init_pop_time += (end - start) / 1e6;
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
    free(fitness);

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

    while (func_count < max_evals) {
        iteration++;
        printf("Iteration %d\n", iteration);

        // Compute mean position on CPU for simplicity
        positions = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population");
        float *mean_pos = (float *)calloc(dim, sizeof(float));
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < dim; j++) {
                mean_pos[j] += positions[i * dim + j];
            }
        }
        for (int j = 0; j < dim; j++) {
            mean_pos[j] /= pop_size;
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, mean_pos_buf, CL_TRUE, 0, dim * sizeof(cl_float), mean_pos, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer mean_pos");
        free(positions);
        free(mean_pos);

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

        // Compute escaping energy
        float E1 = ENERGY_FACTOR * (1.0f - ((float)iteration / (max_evals / pop_size)));
        srand(time(NULL));
        float E0 = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        float escaping_energy = E1 * E0;

        if (fabs(escaping_energy) >= 1.0f) {
            // Exploration phase
            cl_event exploration_event;
            clSetKernelArg(cl_ctx.exploration_kernel, 0, sizeof(cl_mem), &ctx.population);
            clSetKernelArg(cl_ctx.exploration_kernel, 1, sizeof(cl_mem), &ctx.fitness);
            clSetKernelArg(cl_ctx.exploration_kernel, 2, sizeof(cl_mem), &ctx.best_position);
            clSetKernelArg(cl_ctx.exploration_kernel, 3, sizeof(cl_mem), &mean_pos_buf);
            clSetKernelArg(cl_ctx.exploration_kernel, 4, sizeof(cl_mem), &ctx.bounds);
            clSetKernelArg(cl_ctx.exploration_kernel, 5, sizeof(cl_mem), &ctx.random_seeds);
            clSetKernelArg(cl_ctx.exploration_kernel, 6, sizeof(cl_int), &dim);
            clSetKernelArg(cl_ctx.exploration_kernel, 7, sizeof(cl_int), &pop_size);

            err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.exploration_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &exploration_event);
            check_cl_error(err, "clEnqueueNDRangeKernel exploration");
            clFinish(cl_ctx.queue);

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
        } else {
            // Exploitation phase
            cl_event levy_event;
            clSetKernelArg(cl_ctx.levy_flight_kernel, 0, sizeof(cl_mem), &levy_step_buf);
            clSetKernelArg(cl_ctx.levy_flight_kernel, 1, sizeof(cl_mem), &ctx.random_seeds);
            clSetKernelArg(cl_ctx.levy_flight_kernel, 2, sizeof(cl_int), &dim);
            clSetKernelArg(cl_ctx.levy_flight_kernel, 3, sizeof(cl_int), &pop_size);

            err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.levy_flight_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &levy_event);
            check_cl_error(err, "clEnqueueNDRangeKernel levy_flight");
            clFinish(cl_ctx.queue);

            if (levy_event) {
                err = clGetEventProfilingInfo(levy_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(levy_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        prof.levy_flight_time += (end - start) / 1e6;
                    }
                }
                clReleaseEvent(levy_event);
            }

            cl_event exploitation_event;
            clSetKernelArg(cl_ctx.exploitation_kernel, 0, sizeof(cl_mem), &ctx.population);
            clSetKernelArg(cl_ctx.exploitation_kernel, 1, sizeof(cl_mem), &ctx.fitness);
            clSetKernelArg(cl_ctx.exploitation_kernel, 2, sizeof(cl_mem), &ctx.best_position);
            clSetKernelArg(cl_ctx.exploitation_kernel, 3, sizeof(cl_float), &ctx.best_fitness);
            clSetKernelArg(cl_ctx.exploitation_kernel, 4, sizeof(cl_mem), &mean_pos_buf);
            clSetKernelArg(cl_ctx.exploitation_kernel, 5, sizeof(cl_mem), &levy_step_buf);
            clSetKernelArg(cl_ctx.exploitation_kernel, 6, sizeof(cl_float), &escaping_energy);
            clSetKernelArg(cl_ctx.exploitation_kernel, 7, sizeof(cl_mem), &ctx.bounds);
            clSetKernelArg(cl_ctx.exploitation_kernel, 8, sizeof(cl_mem), &ctx.random_seeds);
            clSetKernelArg(cl_ctx.exploitation_kernel, 9, sizeof(cl_int), &dim);
            clSetKernelArg(cl_ctx.exploitation_kernel, 10, sizeof(cl_int), &pop_size);

            err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.exploitation_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &exploitation_event);
            check_cl_error(err, "clEnqueueNDRangeKernel exploitation");
            clFinish(cl_ctx.queue);

            if (exploitation_event) {
                err = clGetEventProfilingInfo(exploitation_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(exploitation_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        prof.exploitation_time += (end - start) / 1e6;
                    }
                }
                clReleaseEvent(exploitation_event);
            }
        }

        // Evaluate population on CPU
        positions = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population");
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
        free(fitness);

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.best_fitness);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_pop_time);
    printf("Exploration Phase: %.3f\n", prof.exploration_time);
    printf("Exploitation Phase: %.3f\n", prof.exploitation_time);
    printf("Levy Flight: %.3f\n", prof.levy_flight_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Exploration Phase: %.3f\n", prof.exploration_time / prof.count);
        printf("Exploitation Phase: %.3f\n", prof.exploitation_time / prof.count);
        printf("Levy MaherFlight: %.3f\n", prof.levy_flight_time / prof.count);
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

    clReleaseMemObject(best_fitness_buf);
    clReleaseMemObject(mean_pos_buf);
    clReleaseMemObject(levy_step_buf);
    HHO_cleanup_context(&ctx, &cl_ctx);
    HHO_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
