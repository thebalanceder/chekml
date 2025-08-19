#include "DRA.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Random number generator (from original DRA.c)
static uint32_t xorshift_state = 1;
void init_xorshift(uint32_t seed) {
    xorshift_state = seed ? seed : (uint32_t)time(NULL);
}

uint32_t xorshift32_dra() {
    uint32_t x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

double rand_double_dra(double min, double max) {
    double r = (double)xorshift32_dra() / UINT32_MAX;
    return min + r * (max - min);
}

// Preprocess kernel source (stub)
char *preprocess_kernel_source(const char *source);

void DRA_init_cl(DRACLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->init_belief_profiles_kernel = clCreateKernel(cl_ctx->program, "initialize_belief_profiles", &err);
    check_cl_error(err, "clCreateKernel initialize_belief_profiles");
    cl_ctx->miracle_operator_kernel = clCreateKernel(cl_ctx->program, "miracle_operator", &err);
    check_cl_error(err, "clCreateKernel miracle_operator");
    cl_ctx->proselytism_operator_kernel = clCreateKernel(cl_ctx->program, "proselytism_operator", &err);
    check_cl_error(err, "clCreateKernel proselytism_operator");
    cl_ctx->reward_penalty_operator_kernel = clCreateKernel(cl_ctx->program, "reward_penalty_operator", &err);
    check_cl_error(err, "clCreateKernel reward_penalty_operator");
    cl_ctx->replacement_operator_kernel = clCreateKernel(cl_ctx->program, "replacement_operator", &err);
    check_cl_error(err, "clCreateKernel replacement_operator");
    cl_ctx->find_min_max_kernel = clCreateKernel(cl_ctx->program, "find_min_max", &err);
    check_cl_error(err, "clCreateKernel find_min_max");
}

void DRA_cleanup_cl(DRACLContext *cl_ctx) {
    if (cl_ctx->find_min_max_kernel) clReleaseKernel(cl_ctx->find_min_max_kernel);
    if (cl_ctx->replacement_operator_kernel) clReleaseKernel(cl_ctx->replacement_operator_kernel);
    if (cl_ctx->reward_penalty_operator_kernel) clReleaseKernel(cl_ctx->reward_penalty_operator_kernel);
    if (cl_ctx->proselytism_operator_kernel) clReleaseKernel(cl_ctx->proselytism_operator_kernel);
    if (cl_ctx->miracle_operator_kernel) clReleaseKernel(cl_ctx->miracle_operator_kernel);
    if (cl_ctx->init_belief_profiles_kernel) clReleaseKernel(cl_ctx->init_belief_profiles_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void DRA_init_context(DRAContext *ctx, Optimizer *opt, DRACLContext *cl_ctx) {
    cl_int err;
    ctx->best_cost = INFINITY;

    int population_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->positions = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer positions");
    ctx->costs = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer costs");
    ctx->best_position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_position");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->min_max_indices = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, 2 * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer min_max_indices");

    // Initialize random seeds
    cl_uint *seeds = (cl_uint *)malloc(population_size * sizeof(cl_uint));
    init_xorshift((uint32_t)time(NULL));
    for (int i = 0; i < population_size; i++) seeds[i] = xorshift32_dra();
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

void DRA_cleanup_context(DRAContext *ctx, DRACLContext *cl_ctx) {
    if (ctx->min_max_indices) clReleaseMemObject(ctx->min_max_indices);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->costs) clReleaseMemObject(ctx->costs);
    if (ctx->positions) clReleaseMemObject(ctx->positions);
}

void DRA_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    printf("Starting GPU DRA optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("DRA.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open DRA.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: DRA.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read DRA.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: DRA.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    DRACLContext cl_ctx = {0};
    DRA_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    DRAContext ctx = {0};
    DRA_init_context(&ctx, opt, &cl_ctx);

    int population_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;
    int stagnation_count = 0;
    float prev_best_cost = INFINITY;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * population_size : DRA_MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_belief_time;
        double miracle_time;
        double proselytism_time;
        double reward_penalty_time;
        double replacement_time;
        double min_max_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;
    cl_mem best_cost_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_cost");

    // Initialize belief profiles
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_belief_profiles_kernel, 0, sizeof(cl_mem), &ctx.positions);
    clSetKernelArg(cl_ctx.init_belief_profiles_kernel, 1, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_belief_profiles_kernel, 2, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_belief_profiles_kernel, 3, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_belief_profiles_kernel, 4, sizeof(cl_int), &population_size);
    size_t global_work_size = population_size;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.init_belief_profiles_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
    check_cl_error(err, "clEnqueueNDRangeKernel init_belief_profiles");
    clFinish(cl_ctx.queue);

    cl_ulong start, end;
    if (init_event) {
        err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.init_belief_time += (end - start) / 1e6;
            }
        }
        clReleaseEvent(init_event);
    }

    // Evaluate initial population on CPU
    cl_event read_event, write_event;
    float *positions = (float *)malloc(population_size * dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer positions");
    float *costs = (float *)malloc(population_size * sizeof(float));
    for (int i = 0; i < population_size; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
        costs[i] = (float)objective_function(pos);
        costs[i] = isnan(costs[i]) ? INFINITY : costs[i];
        free(pos);
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, population_size * sizeof(cl_float), costs, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer costs");
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

    func_count += population_size;
    float best_cost = costs[0];
    int best_idx = 0;
    for (int i = 1; i < population_size; i++) {
        if (costs[i] < best_cost) {
            best_cost = costs[i];
            best_idx = i;
        }
    }
    ctx.best_cost = best_cost;

    positions = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
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

    err = clEnqueueWriteBuffer(cl_ctx.queue, best_cost_buf, CL_TRUE, 0, sizeof(cl_float), &best_cost, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer best_cost");

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
        float miracle_rate = (float)(rand_double_dra(0.0, 1.0) * (1.0f - ((float)iteration / opt->max_iter * 2.0f)) * rand_double_dra(0.0, 1.0));

        // Find min/max indices
        cl_event min_max_event;
        clSetKernelArg(cl_ctx.find_min_max_kernel, 0, sizeof(cl_mem), &ctx.costs);
        clSetKernelArg(cl_ctx.find_min_max_kernel, 1, sizeof(cl_mem), &ctx.min_max_indices);
        clSetKernelArg(cl_ctx.find_min_max_kernel, 2, sizeof(cl_int), &population_size);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.find_min_max_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &min_max_event);
        check_cl_error(err, "clEnqueueNDRangeKernel find_min_max");
        clFinish(cl_ctx.queue);

        if (min_max_event) {
            err = clGetEventProfilingInfo(min_max_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(min_max_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.min_max_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(min_max_event);
        }

        // Create new follower
        float *new_follower = (float *)malloc(dim * sizeof(float));
        for (int j = 0; j < dim; j++) {
            new_follower[j] = (float)rand_double_dra(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        float new_fitness;
        double new_fitness_d = objective_function((double *)new_follower);
        new_fitness = (float)new_fitness_d;
        new_fitness = isnan(new_fitness) ? INFINITY : new_fitness;
        func_count++;

        // Belief Profile Consideration
        if (rand_double_dra(0.0, 1.0) <= DRA_BELIEF_PROFILE_RATE) {
            int rand_idx = (int)(rand_double_dra(0, population_size));
            int rand_dim = (int)(rand_double_dra(0, dim));
            positions = (float *)malloc(population_size * dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer positions");
            new_follower[rand_dim] = positions[rand_idx * dim + rand_dim];
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
        }

        // Exploration or Exploitation
        if (rand_double_dra(0.0, 1.0) <= miracle_rate) {
            // Miracle operator
            cl_event miracle_event;
            clSetKernelArg(cl_ctx.miracle_operator_kernel, 0, sizeof(cl_mem), &ctx.positions);
            clSetKernelArg(cl_ctx.miracle_operator_kernel, 1, sizeof(cl_mem), &ctx.costs);
            clSetKernelArg(cl_ctx.miracle_operator_kernel, 2, sizeof(cl_mem), &ctx.bounds);
            clSetKernelArg(cl_ctx.miracle_operator_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
            clSetKernelArg(cl_ctx.miracle_operator_kernel, 4, sizeof(cl_int), &dim);
            clSetKernelArg(cl_ctx.miracle_operator_kernel, 5, sizeof(cl_int), &population_size);
            err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.miracle_operator_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &miracle_event);
            check_cl_error(err, "clEnqueueNDRangeKernel miracle_operator");
            clFinish(cl_ctx.queue);

            if (miracle_event) {
                err = clGetEventProfilingInfo(miracle_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(miracle_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        prof.miracle_time += (end - start) / 1e6;
                    }
                }
                clReleaseEvent(miracle_event);
            }

            // Evaluate updated population
            positions = (float *)malloc(population_size * dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer positions");
            for (int i = 0; i < population_size; i++) {
                double *pos = (double *)malloc(dim * sizeof(double));
                for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
                costs[i] = (float)objective_function(pos);
                costs[i] = isnan(costs[i]) ? INFINITY : costs[i];
                free(pos);
            }
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, population_size * sizeof(cl_float), costs, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer costs");
            free(positions);
            func_count += population_size;

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
        } else {
            // Exploitation: update new_follower
            int min_idx;
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.min_max_indices, CL_TRUE, 0, sizeof(cl_int), &min_idx, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer min_idx");
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

            positions = (float *)malloc(dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, min_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer min_pos");
            for (int j = 0; j < dim; j++) {
                new_follower[j] = positions[j] * ((float)rand_double_dra(0.0, 1.0) - sin((float)rand_double_dra(0.0, 1.0)));
                new_follower[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_follower[j]));
            }
            free(positions);
            new_fitness_d = objective_function((double *)new_follower);
            new_fitness = (float)new_fitness_d;
            new_fitness = isnan(new_fitness) ? INFINITY : new_fitness;
            func_count++;

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

            // Proselytism operator
            cl_event proselytism_event;
            clSetKernelArg(cl_ctx.proselytism_operator_kernel, 0, sizeof(cl_mem), &ctx.positions);
            clSetKernelArg(cl_ctx.proselytism_operator_kernel, 1, sizeof(cl_mem), &ctx.costs);
            clSetKernelArg(cl_ctx.proselytism_operator_kernel, 2, sizeof(cl_mem), &ctx.bounds);
            clSetKernelArg(cl_ctx.proselytism_operator_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
            clSetKernelArg(cl_ctx.proselytism_operator_kernel, 4, sizeof(cl_int), &min_idx);
            clSetKernelArg(cl_ctx.proselytism_operator_kernel, 5, sizeof(cl_int), &dim);
            clSetKernelArg(cl_ctx.proselytism_operator_kernel, 6, sizeof(cl_int), &population_size);
            err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.proselytism_operator_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &proselytism_event);
            check_cl_error(err, "clEnqueueNDRangeKernel proselytism_operator");
            clFinish(cl_ctx.queue);

            if (proselytism_event) {
                err = clGetEventProfilingInfo(proselytism_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(proselytism_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        prof.proselytism_time += (end - start) / 1e6;
                    }
                }
                clReleaseEvent(proselytism_event);
            }

            // Evaluate updated population
            positions = (float *)malloc(population_size * dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer positions");
            for (int i = 0; i < population_size; i++) {
                double *pos = (double *)malloc(dim * sizeof(double));
                for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
                costs[i] = (float)objective_function(pos);
                costs[i] = isnan(costs[i]) ? INFINITY : costs[i];
                free(pos);
            }
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, population_size * sizeof(cl_float), costs, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer costs");
            free(positions);
            func_count += population_size;

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

        // Update worst solution
        int worst_idx;
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.min_max_indices, CL_TRUE, sizeof(cl_int), sizeof(cl_int), &worst_idx, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer worst_idx");
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

        if (new_fitness < costs[worst_idx]) {
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, worst_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), new_follower, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer worst_pos");
            costs[worst_idx] = new_fitness;
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, worst_idx * sizeof(cl_float), sizeof(cl_float), &new_fitness, 0, NULL, NULL);
            check_cl_error(err, "clEnqueueWriteBuffer worst_cost");
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
        free(new_follower);

        // Reward-penalty operator
        cl_event reward_event;
        clSetKernelArg(cl_ctx.reward_penalty_operator_kernel, 0, sizeof(cl_mem), &ctx.positions);
        clSetKernelArg(cl_ctx.reward_penalty_operator_kernel, 1, sizeof(cl_mem), &ctx.costs);
        clSetKernelArg(cl_ctx.reward_penalty_operator_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.reward_penalty_operator_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.reward_penalty_operator_kernel, 4, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.reward_penalty_operator_kernel, 5, sizeof(cl_int), &population_size);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.reward_penalty_operator_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &reward_event);
        check_cl_error(err, "clEnqueueNDRangeKernel reward_penalty_operator");
        clFinish(cl_ctx.queue);

        if (reward_event) {
            err = clGetEventProfilingInfo(reward_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(reward_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.reward_penalty_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(reward_event);
        }

        // Evaluate updated population
        positions = (float *)malloc(population_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer positions");
        for (int i = 0; i < population_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            costs[i] = (float)objective_function(pos);
            costs[i] = isnan(costs[i]) ? INFINITY : costs[i];
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, population_size * sizeof(cl_float), costs, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer costs");
        free(positions);
        func_count += population_size;

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

        // Replacement operator
        cl_event replacement_event;
        int num_groups = DRA_NUM_GROUPS < population_size ? DRA_NUM_GROUPS : population_size;
        clSetKernelArg(cl_ctx.replacement_operator_kernel, 0, sizeof(cl_mem), &ctx.positions);
        clSetKernelArg(cl_ctx.replacement_operator_kernel, 1, sizeof(cl_mem), &ctx.costs);
        clSetKernelArg(cl_ctx.replacement_operator_kernel, 2, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.replacement_operator_kernel, 3, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.replacement_operator_kernel, 4, sizeof(cl_int), &population_size);
        clSetKernelArg(cl_ctx.replacement_operator_kernel, 5, sizeof(cl_int), &num_groups);
        size_t replacement_work_size = num_groups;
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.replacement_operator_kernel, 1, NULL, &replacement_work_size, NULL, 0, NULL, &replacement_event);
        check_cl_error(err, "clEnqueueNDRangeKernel replacement_operator");
        clFinish(cl_ctx.queue);

        if (replacement_event) {
            err = clGetEventProfilingInfo(replacement_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(replacement_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.replacement_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(replacement_event);
        }

        // Evaluate updated population
        positions = (float *)malloc(population_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer positions");
        for (int i = 0; i < population_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            costs[i] = (float)objective_function(pos);
            costs[i] = isnan(costs[i]) ? INFINITY : costs[i];
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, population_size * sizeof(cl_float), costs, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer costs");
        free(positions);
        func_count += population_size;

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
        float new_best = costs[0];
        int new_best_idx = 0;
        for (int i = 1; i < population_size; i++) {
            if (costs[i] < new_best) {
                new_best = costs[i];
                new_best_idx = i;
            }
        }
        if (new_best < ctx.best_cost) {
            ctx.best_cost = new_best;
            best_idx = new_best_idx;
            positions = (float *)malloc(dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
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

            err = clEnqueueWriteBuffer(cl_ctx.queue, best_cost_buf, CL_TRUE, 0, sizeof(cl_float), &new_best, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer best_cost");

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

            // Check for convergence
            if (fabs(prev_best_cost - new_best) < DRA_CONVERGENCE_TOL) {
                stagnation_count++;
                if (stagnation_count >= DRA_STAGNATION_THRESHOLD) {
                    printf("Convergence reached, stopping early\n");
                    break;
                }
            } else {
                stagnation_count = 0;
            }
            prev_best_cost = new_best;
        }

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Cost = %f\n", iteration, func_count, ctx.best_cost);
    }

    // Copy best solution to optimizer
    float *best_pos = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer final best_pos");
    opt->best_solution.fitness = ctx.best_cost;
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

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Belief Profiles: %.3f\n", prof.init_belief_time);
    printf("Miracle Operator: %.3f\n", prof.miracle_time);
    printf("Proselytism Operator: %.3f\n", prof.proselytism_time);
    printf("Reward-Penalty Operator: %.3f\n", prof.reward_penalty_time);
    printf("Replacement Operator: %.3f\n", prof.replacement_time);
    printf("Find Min/Max: %.3f\n", prof.min_max_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Miracle Operator: %.3f\n", prof.miracle_time / prof.count);
        printf("Proselytism Operator: %.3f\n", prof.proselytism_time / prof.count);
        printf("Reward-Penalty Operator: %.3f\n", prof.reward_penalty_time / prof.count);
        printf("Replacement Operator: %.3f\n", prof.replacement_time / prof.count);
        printf("Find Min/Max: %.3f\n", prof.min_max_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Cleanup
    free(costs);
    clReleaseMemObject(best_cost_buf);
    DRA_cleanup_context(&ctx, &cl_ctx);
    DRA_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
