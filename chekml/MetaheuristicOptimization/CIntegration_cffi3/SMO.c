#include "SMO.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (placeholder, can be expanded if needed)
char *preprocess_kernel_source(const char *source);

void SMO_init_cl(SMOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->local_leader_kernel = clCreateKernel(cl_ctx->program, "local_leader_phase", &err);
    check_cl_error(err, "clCreateKernel local_leader");
    cl_ctx->global_leader_kernel = clCreateKernel(cl_ctx->program, "global_leader_phase", &err);
    check_cl_error(err, "clCreateKernel global_leader");
    cl_ctx->bhc_kernel = clCreateKernel(cl_ctx->program, "beta_hill_climbing", &err);
    check_cl_error(err, "clCreateKernel bhc");
}

void SMO_cleanup_cl(SMOCLContext *cl_ctx) {
    if (cl_ctx->bhc_kernel) clReleaseKernel(cl_ctx->bhc_kernel);
    if (cl_ctx->global_leader_kernel) clReleaseKernel(cl_ctx->global_leader_kernel);
    if (cl_ctx->local_leader_kernel) clReleaseKernel(cl_ctx->local_leader_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void SMO_init_context(SMOContext *ctx, Optimizer *opt, SMOCLContext *cl_ctx) {
    cl_int err;
    ctx->global_leader_fitness = INFINITY;

    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->population = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer population");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->group_ids = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer group_ids");
    ctx->group_leaders = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, SMO_MAX_GROUPS * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer group_leaders");
    ctx->group_leader_fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, SMO_MAX_GROUPS * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer group_leader_fitness");
    ctx->global_leader = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer global_leader");

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

    // Initialize group IDs (all in group 0 initially)
    cl_int *group_ids = (cl_int *)malloc(pop_size * sizeof(cl_int));
    for (int i = 0; i < pop_size; i++) group_ids[i] = 0;
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->group_ids, CL_TRUE, 0, pop_size * sizeof(cl_int), group_ids, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer group_ids");
    free(group_ids);
}

void SMO_cleanup_context(SMOContext *ctx, SMOCLContext *cl_ctx) {
    if (ctx->global_leader) clReleaseMemObject(ctx->global_leader);
    if (ctx->group_leader_fitness) clReleaseMemObject(ctx->group_leader_fitness);
    if (ctx->group_leaders) clReleaseMemObject(ctx->group_leaders);
    if (ctx->group_ids) clReleaseMemObject(ctx->group_ids);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

void SMO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Spider Monkey Optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("SMO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open SMO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: SMO.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read SMO.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: SMO.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    SMOCLContext cl_ctx = {0};
    SMO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    SMOContext ctx = {0};
    SMO_init_context(&ctx, opt, &cl_ctx);

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
        double local_leader_time;
        double global_leader_time;
        double bhc_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    // Declare err for OpenCL calls
    cl_int err;

    // Initialize population
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_pop_kernel, 0, sizeof(cl_mem), &ctx.population);
    clSetKernelArg(cl_ctx.init_pop_kernel, 1, sizeof(cl_mem), &ctx.fitness);
    clSetKernelArg(cl_ctx.init_pop_kernel, 2, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 4, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_pop_kernel, 5, sizeof(cl_int), &pop_size);

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

    // Initialize group and global leaders
    float best_fitness = fitness[0];
    int best_idx = 0;
    for (int i = 1; i < pop_size; i++) {
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            best_idx = i;
        }
    }
    ctx.global_leader_fitness = best_fitness;
    positions = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer global_leader");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.global_leader, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer global_leader");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.group_leaders, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer group_leaders");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.group_leader_fitness, CL_TRUE, 0, sizeof(cl_float), &best_fitness, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer group_leader_fitness");
    free(positions);
    free(fitness);

    // Group management structures
    typedef struct {
        int *members;
        int size;
        float *leader_position;
        float leader_fitness;
        int leader_count;
    } Group;
    Group *groups = (Group *)malloc(SMO_MAX_GROUPS * sizeof(Group));
    int num_groups = 1;
    groups[0].members = (int *)malloc(pop_size * sizeof(int));
    groups[0].size = pop_size;
    for (int i = 0; i < pop_size; i++) groups[0].members[i] = i;
    groups[0].leader_position = (float *)malloc(dim * sizeof(float));
    memcpy(groups[0].leader_position, positions, dim * sizeof(float));
    groups[0].leader_fitness = best_fitness;
    groups[0].leader_count = 0;

    // Main optimization loop
    while (func_count < max_evals) {
        iteration++;
        printf("Iteration %d\n", iteration);

        // Local Leader Phase
        cl_event local_leader_event;
        clSetKernelArg(cl_ctx.local_leader_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.local_leader_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.local_leader_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.local_leader_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.local_leader_kernel, 4, sizeof(cl_mem), &ctx.group_ids);
        clSetKernelArg(cl_ctx.local_leader_kernel, 5, sizeof(cl_mem), &ctx.group_leaders);
        clSetKernelArg(cl_ctx.local_leader_kernel, 6, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.local_leader_kernel, 7, sizeof(cl_int), &pop_size);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.local_leader_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &local_leader_event);
        check_cl_error(err, "clEnqueueNDRangeKernel local_leader");
        clFinish(cl_ctx.queue);

        double local_leader_time = 0.0;
        if (local_leader_event) {
            err = clGetEventProfilingInfo(local_leader_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(local_leader_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    local_leader_time = (end - start) / 1e6;
                    prof.local_leader_time += local_leader_time;
                }
            }
            clReleaseEvent(local_leader_event);
        }

        // Global Leader Phase
        cl_event global_leader_event;
        float max_fitness = -INFINITY;
        fitness = (float *)malloc(pop_size * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer fitness");
        for (int i = 0; i < pop_size; i++) {
            if (fitness[i] > max_fitness) max_fitness = fitness[i];
        }
        clSetKernelArg(cl_ctx.global_leader_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.global_leader_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.global_leader_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.global_leader_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.global_leader_kernel, 4, sizeof(cl_mem), &ctx.group_ids);
        clSetKernelArg(cl_ctx.global_leader_kernel, 5, sizeof(cl_mem), &ctx.global_leader);
        clSetKernelArg(cl_ctx.global_leader_kernel, 6, sizeof(cl_float), &max_fitness);
        clSetKernelArg(cl_ctx.global_leader_kernel, 7, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.global_leader_kernel, 8, sizeof(cl_int), &pop_size);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.global_leader_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &global_leader_event);
        check_cl_error(err, "clEnqueueNDRangeKernel global_leader");
        clFinish(cl_ctx.queue);

        double global_leader_time = 0.0;
        if (global_leader_event) {
            err = clGetEventProfilingInfo(global_leader_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(global_leader_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    global_leader_time = (end - start) / 1e6;
                    prof.global_leader_time += global_leader_time;
                }
            }
            clReleaseEvent(global_leader_event);
        }

        // Beta-Hill Climbing
        cl_event bhc_event;
        clSetKernelArg(cl_ctx.bhc_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.bhc_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.bhc_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.bhc_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.bhc_kernel, 4, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.bhc_kernel, 5, sizeof(cl_int), &pop_size);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.bhc_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &bhc_event);
        check_cl_error(err, "clEnqueueNDRangeKernel bhc");
        clFinish(cl_ctx.queue);

        double bhc_time = 0.0;
        if (bhc_event) {
            err = clGetEventProfilingInfo(bhc_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(bhc_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    bhc_time = (end - start) / 1e6;
                    prof.bhc_time += bhc_time;
                }
            }
            clReleaseEvent(bhc_event);
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

        // Local Leader Decision
        for (int g = 0; g < num_groups; g++) {
            int best_idx = groups[g].members[0];
            float best_fitness = fitness[best_idx];
            for (int i = 1; i < groups[g].size; i++) {
                int idx = groups[g].members[i];
                if (fitness[idx] < best_fitness) {
                    best_idx = idx;
                    best_fitness = fitness[idx];
                }
            }
            if (best_fitness < groups[g].leader_fitness) {
                groups[g].leader_fitness = best_fitness;
                positions = (float *)malloc(dim * sizeof(float));
                err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, NULL);
                check_cl_error(err, "clEnqueueReadBuffer group_leader");
                memcpy(groups[g].leader_position, positions, dim * sizeof(float));
                err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.group_leaders, CL_TRUE, g * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, NULL);
                check_cl_error(err, "clEnqueueWriteBuffer group_leaders");
                err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.group_leader_fitness, CL_TRUE, g * sizeof(cl_float), sizeof(cl_float), &best_fitness, 0, NULL, NULL);
                check_cl_error(err, "clEnqueueWriteBuffer group_leader_fitness");
                free(positions);
                groups[g].leader_count = 0;
            } else {
                groups[g].leader_count++;
                if (groups[g].leader_count > SMO_LOCAL_LEADER_LIMIT) {
                    // Trigger beta-hill climbing in next iteration
                    groups[g].leader_count = 0;
                }
            }
        }

        // Global Leader Decision
        best_idx = 0;
        best_fitness = fitness[0];
        for (int i = 1; i < pop_size; i++) {
            if (fitness[i] < best_fitness) {
                best_idx = i;
                best_fitness = fitness[i];
            }
        }
        if (best_fitness < ctx.global_leader_fitness) {
            ctx.global_leader_fitness = best_fitness;
            positions = (float *)malloc(dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer global_leader");
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.global_leader, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer global_leader");
            free(positions);
            groups[0].leader_count = 0;
        } else {
            groups[0].leader_count++;
        }

        if (groups[0].leader_count > SMO_GLOBAL_LEADER_LIMIT) {
            groups[0].leader_count = 0;
            if (num_groups < SMO_MAX_GROUPS) {
                int largest_idx = 0;
                for (int g = 1; g < num_groups; g++) {
                    if (groups[g].size > groups[largest_idx].size) largest_idx = g;
                }
                if (groups[largest_idx].size > 1) {
                    // Split largest group
                    Group *new_groups = (Group *)malloc((num_groups + 1) * sizeof(Group));
                    memcpy(new_groups, groups, num_groups * sizeof(Group));
                    new_groups[num_groups].members = (int *)malloc(groups[largest_idx].size * sizeof(int));
                    new_groups[num_groups].leader_position = (float *)malloc(dim * sizeof(float));
                    new_groups[num_groups].leader_fitness = INFINITY;
                    new_groups[num_groups].leader_count = 0;

                    int split_point = groups[largest_idx].size / 2;
                    new_groups[num_groups].size = groups[largest_idx].size - split_point;
                    new_groups[largest_idx].size = split_point;
                    memcpy(new_groups[num_groups].members, groups[largest_idx].members + split_point, new_groups[num_groups].size * sizeof(int));
                    new_groups[num_groups].leader_fitness = fitness[new_groups[num_groups].members[0]];
                    positions = (float *)malloc(dim * sizeof(float));
                    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, new_groups[num_groups].members[0] * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, NULL);
                    check_cl_error(err, "clEnqueueReadBuffer new_group_leader");
                    memcpy(new_groups[num_groups].leader_position, positions, dim * sizeof(float));
                    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.group_leaders, CL_TRUE, num_groups * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, NULL);
                    check_cl_error(err, "clEnqueueWriteBuffer new_group_leaders");
                    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.group_leader_fitness, CL_TRUE, num_groups * sizeof(cl_float), sizeof(cl_float), &new_groups[num_groups].leader_fitness, 0, NULL, NULL);
                    check_cl_error(err, "clEnqueueWriteBuffer new_group_leader_fitness");
                    free(positions);

                    // Update group IDs
                    cl_int *group_ids = (cl_int *)malloc(pop_size * sizeof(cl_int));
                    for (int i = 0; i < pop_size; i++) group_ids[i] = -1;
                    for (int g = 0; g <= num_groups; g++) {
                        for (int i = 0; i < new_groups[g].size; i++) {
                            group_ids[new_groups[g].members[i]] = g;
                        }
                    }
                    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.group_ids, CL_TRUE, 0, pop_size * sizeof(cl_int), group_ids, 0, NULL, NULL);
                    check_cl_error(err, "clEnqueueWriteBuffer group_ids");
                    free(group_ids);

                    free(groups);
                    groups = new_groups;
                    num_groups++;
                }
            } else {
                // Merge all groups
                Group *new_groups = (Group *)malloc(sizeof(Group));
                new_groups[0].size = pop_size;
                new_groups[0].members = (int *)malloc(pop_size * sizeof(int));
                for (int i = 0; i < pop_size; i++) new_groups[0].members[i] = i;
                new_groups[0].leader_position = (float *)malloc(dim * sizeof(float));
                positions = (float *)malloc(dim * sizeof(float));
                err = clEnqueueReadBuffer(cl_ctx.queue, ctx.global_leader, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, NULL);
                check_cl_error(err, "clEnqueueReadBuffer merge_leader");
                memcpy(new_groups[0].leader_position, positions, dim * sizeof(float));
                new_groups[0].leader_fitness = ctx.global_leader_fitness;
                new_groups[0].leader_count = 0;

                cl_int *group_ids = (cl_int *)malloc(pop_size * sizeof(cl_int));
                for (int i = 0; i < pop_size; i++) group_ids[i] = 0;
                err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.group_ids, CL_TRUE, 0, pop_size * sizeof(cl_int), group_ids, 0, NULL, NULL);
                check_cl_error(err, "clEnqueueWriteBuffer group_ids");
                free(group_ids);

                for (int g = 0; g < num_groups; g++) {
                    free(groups[g].members);
                    free(groups[g].leader_position);
                }
                free(groups);
                groups = new_groups;
                num_groups = 1;
            }
        }

        free(fitness);
        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.global_leader_fitness);
        printf("Profiling (ms): Local Leader = %.3f, Global Leader = %.3f, BHC = %.3f\n", local_leader_time, global_leader_time, bhc_time);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_pop_time);
    printf("Local Leader Phase: %.3f\n", prof.local_leader_time);
    printf("Global Leader Phase: %.3f\n", prof.global_leader_time);
    printf("Beta-Hill Climbing: %.3f\n", prof.bhc_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Local Leader Phase: %.3f\n", prof.local_leader_time / prof.count);
        printf("Global Leader Phase: %.3f\n", prof.global_leader_time / prof.count);
        printf("Beta-Hill Climbing: %.3f\n", prof.bhc_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Copy best solution to optimizer
    float *best_pos = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.global_leader, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer final global_leader");
    opt->best_solution.fitness = ctx.global_leader_fitness;
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
    for (int g = 0; g < num_groups; g++) {
        free(groups[g].members);
        free(groups[g].leader_position);
    }
    free(groups);
    SMO_cleanup_context(&ctx, &cl_ctx);
    SMO_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
