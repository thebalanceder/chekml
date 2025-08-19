#include "MfA.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source to fix address space issues
char *preprocess_kernel_source(const char *source);

void MfA_init_cl(MfACLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->update_males_kernel = clCreateKernel(cl_ctx->program, "update_males", &err);
    check_cl_error(err, "clCreateKernel update_males");
    cl_ctx->update_females_kernel = clCreateKernel(cl_ctx->program, "update_females", &err);
    check_cl_error(err, "clCreateKernel update_females");
    cl_ctx->mating_kernel = clCreateKernel(cl_ctx->program, "mating_phase", &err);
    check_cl_error(err, "clCreateKernel mating_phase");
    cl_ctx->mutation_kernel = clCreateKernel(cl_ctx->program, "mutation_phase", &err);
    check_cl_error(err, "clCreateKernel mutation_phase");
}

void MfA_cleanup_cl(MfACLContext *cl_ctx) {
    if (cl_ctx->mutation_kernel) clReleaseKernel(cl_ctx->mutation_kernel);
    if (cl_ctx->mating_kernel) clReleaseKernel(cl_ctx->mating_kernel);
    if (cl_ctx->update_females_kernel) clReleaseKernel(cl_ctx->update_females_kernel);
    if (cl_ctx->update_males_kernel) clReleaseKernel(cl_ctx->update_males_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void MfA_init_context(MfAContext *ctx, Optimizer *opt, MfACLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;

    int half_pop = opt->population_size / 2;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->male_population = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, half_pop * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer male_population");
    ctx->female_population = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, half_pop * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer female_population");
    ctx->male_fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, half_pop * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer male_fitness");
    ctx->female_fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, half_pop * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer female_fitness");
    ctx->male_velocities = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, half_pop * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer male_velocities");
    ctx->female_velocities = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, half_pop * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer female_velocities");
    ctx->best_male_positions = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, half_pop * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_male_positions");
    ctx->best_position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_position");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, opt->population_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");

    // Initialize random seeds
    cl_uint *seeds = (cl_uint *)malloc(opt->population_size * sizeof(cl_uint));
    srand(time(NULL));
    for (int i = 0; i < opt->population_size; i++) seeds[i] = rand();
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->random_seeds, CL_TRUE, 0, opt->population_size * sizeof(cl_uint), seeds, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer seeds");
    free(seeds);

    // Initialize bounds
    float *bounds_float = (float *)malloc(dim * 2 * sizeof(float));
    for (int i = 0; i < dim * 2; i++) bounds_float[i] = (float)opt->bounds[i];
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->bounds, CL_TRUE, 0, dim * 2 * sizeof(cl_float), bounds_float, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer bounds");
    free(bounds_float);
}

void MfA_cleanup_context(MfAContext *ctx, MfACLContext *cl_ctx) {
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->best_male_positions) clReleaseMemObject(ctx->best_male_positions);
    if (ctx->female_velocities) clReleaseMemObject(ctx->female_velocities);
    if (ctx->male_velocities) clReleaseMemObject(ctx->male_velocities);
    if (ctx->female_fitness) clReleaseMemObject(ctx->female_fitness);
    if (ctx->male_fitness) clReleaseMemObject(ctx->male_fitness);
    if (ctx->female_population) clReleaseMemObject(ctx->female_population);
    if (ctx->male_population) clReleaseMemObject(ctx->male_population);
}

void MfA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Mayfly Algorithm optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("MfA.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open MfA.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: MfA.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read MfA.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: MfA.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    MfACLContext cl_ctx = {0};
    MfA_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    MfAContext ctx = {0};
    MfA_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int half_pop = pop_size / 2;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * pop_size : MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_pop_time;
        double update_males_time;
        double update_females_time;
        double mating_time;
        double mutation_time;
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
    clSetKernelArg(cl_ctx.init_pop_kernel, 0, sizeof(cl_mem), &ctx.male_population);
    clSetKernelArg(cl_ctx.init_pop_kernel, 1, sizeof(cl_mem), &ctx.female_population);
    clSetKernelArg(cl_ctx.init_pop_kernel, 2, sizeof(cl_mem), &ctx.male_fitness);
    clSetKernelArg(cl_ctx.init_pop_kernel, 3, sizeof(cl_mem), &ctx.female_fitness);
    clSetKernelArg(cl_ctx.init_pop_kernel, 4, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 5, sizeof(cl_mem), &ctx.best_position);
    clSetKernelArg(cl_ctx.init_pop_kernel, 6, sizeof(cl_mem), &best_fitness_buf);
    clSetKernelArg(cl_ctx.init_pop_kernel, 7, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 8, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_pop_kernel, 9, sizeof(cl_int), &half_pop);

    size_t global_work_size = half_pop;
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

    // Evaluate initial population on CPU
    cl_event read_event, write_event;
    float *male_positions = (float *)malloc(half_pop * dim * sizeof(float));
    float *female_positions = (float *)malloc(half_pop * dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.male_population, CL_TRUE, 0, half_pop * dim * sizeof(cl_float), male_positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer male_population");
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.female_population, CL_TRUE, 0, half_pop * dim * sizeof(cl_float), female_positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueReadBuffer female_population");

    float *male_fitness = (float *)malloc(half_pop * sizeof(float));
    float *female_fitness = (float *)malloc(half_pop * sizeof(float));
    float best_fitness = INFINITY;
    int best_idx = 0;
    int is_male = 1;

    for (int i = 0; i < half_pop; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = male_positions[i * dim + j];
        male_fitness[i] = (float)objective_function(pos);
        if (male_fitness[i] < best_fitness) {
            best_fitness = male_fitness[i];
            best_idx = i;
            is_male = 1;
        }
        free(pos);
    }
    for (int i = 0; i < half_pop; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = female_positions[i * dim + j];
        female_fitness[i] = (float)objective_function(pos);
        if (female_fitness[i] < best_fitness) {
            best_fitness = female_fitness[i];
            best_idx = i;
            is_male = 0;
        }
        free(pos);
    }

    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.male_fitness, CL_TRUE, 0, half_pop * sizeof(cl_float), male_fitness, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer male_fitness");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.female_fitness, CL_TRUE, 0, half_pop * sizeof(cl_float), female_fitness, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer female_fitness");

    ctx.best_fitness = best_fitness;
    float *best_pos = (float *)malloc(dim * sizeof(float));
    if (is_male) {
        for (int j = 0; j < dim; j++) best_pos[j] = male_positions[best_idx * dim + j];
    } else {
        for (int j = 0; j < dim; j++) best_pos[j] = female_positions[best_idx * dim + j];
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer best_position");
    err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &best_fitness, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer best_fitness");

    // Initialize best male positions
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_male_positions, CL_TRUE, 0, half_pop * dim * sizeof(cl_float), male_positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer best_male_positions");

    free(male_positions);
    free(female_positions);
    free(male_fitness);
    free(female_fitness);
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

    // Main optimization loop
    float inertia_weight = MFA_INERTIA_WEIGHT;
    float nuptial_dance = MFA_NUPTIAL_DANCE;
    float random_flight = MFA_RANDOM_FLIGHT;
    float mutation_rate = MFA_MUTATION_RATE;

    while (func_count < max_evals) {
        iteration++;
        printf("Iteration %d\n", iteration);

        // Update males
        cl_event update_males_event;
        cl_float vel_max = 0.1f * (opt->bounds[1] - opt->bounds[0]);
        cl_float vel_min = -vel_max;
        clSetKernelArg(cl_ctx.update_males_kernel, 0, sizeof(cl_mem), &ctx.male_population);
        clSetKernelArg(cl_ctx.update_males_kernel, 1, sizeof(cl_mem), &ctx.male_fitness);
        clSetKernelArg(cl_ctx.update_males_kernel, 2, sizeof(cl_mem), &ctx.male_velocities);
        clSetKernelArg(cl_ctx.update_males_kernel, 3, sizeof(cl_mem), &ctx.best_male_positions);
        clSetKernelArg(cl_ctx.update_males_kernel, 4, sizeof(cl_mem), &ctx.best_position);
        clSetKernelArg(cl_ctx.update_males_kernel, 5, sizeof(cl_mem), &best_fitness_buf);
        clSetKernelArg(cl_ctx.update_males_kernel, 6, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.update_males_kernel, 7, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.update_males_kernel, 8, sizeof(cl_float), &inertia_weight);
        clSetKernelArg(cl_ctx.update_males_kernel, 9, sizeof(cl_float), &vel_max);
        clSetKernelArg(cl_ctx.update_males_kernel, 10, sizeof(cl_float), &vel_min);
        clSetKernelArg(cl_ctx.update_males_kernel, 11, sizeof(cl_float), &nuptial_dance);
        clSetKernelArg(cl_ctx.update_males_kernel, 12, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.update_males_kernel, 13, sizeof(cl_int), &half_pop);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_males_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &update_males_event);
        check_cl_error(err, "clEnqueueNDRangeKernel update_males");
        clFinish(cl_ctx.queue);

        double update_males_time = 0.0;
        if (update_males_event) {
            err = clGetEventProfilingInfo(update_males_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(update_males_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    update_males_time = (end - start) / 1e6;
                    prof.update_males_time += update_males_time;
                }
            }
            clReleaseEvent(update_males_event);
        }

        // Update females
        cl_event update_females_event;
        clSetKernelArg(cl_ctx.update_females_kernel, 0, sizeof(cl_mem), &ctx.female_population);
        clSetKernelArg(cl_ctx.update_females_kernel, 1, sizeof(cl_mem), &ctx.female_fitness);
        clSetKernelArg(cl_ctx.update_females_kernel, 2, sizeof(cl_mem), &ctx.female_velocities);
        clSetKernelArg(cl_ctx.update_females_kernel, 3, sizeof(cl_mem), &ctx.male_population);
        clSetKernelArg(cl_ctx.update_females_kernel, 4, sizeof(cl_mem), &ctx.male_fitness);
        clSetKernelArg(cl_ctx.update_females_kernel, 5, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.update_females_kernel, 6, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.update_females_kernel, 7, sizeof(cl_float), &inertia_weight);
        clSetKernelArg(cl_ctx.update_females_kernel, 8, sizeof(cl_float), &random_flight);
        clSetKernelArg(cl_ctx.update_females_kernel, 9, sizeof(cl_float), &vel_max);
        clSetKernelArg(cl_ctx.update_females_kernel, 10, sizeof(cl_float), &vel_min);
        clSetKernelArg(cl_ctx.update_females_kernel, 11, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.update_females_kernel, 12, sizeof(cl_int), &half_pop);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_females_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &update_females_event);
        check_cl_error(err, "clEnqueueNDRangeKernel update_females");
        clFinish(cl_ctx.queue);

        double update_females_time = 0.0;
        if (update_females_event) {
            err = clGetEventProfilingInfo(update_females_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(update_females_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    update_females_time = (end - start) / 1e6;
                    prof.update_females_time += update_females_time;
                }
            }
            clReleaseEvent(update_females_event);
        }

        // Mating phase
        cl_event mating_event;
        cl_mem offspring_population = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, MFA_NUM_OFFSPRING * dim * sizeof(cl_float), NULL, &err);
        check_cl_error(err, "clCreateBuffer offspring_population");
        cl_mem offspring_fitness_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, MFA_NUM_OFFSPRING * sizeof(cl_float), NULL, &err);
        check_cl_error(err, "clCreateBuffer offspring_fitness");

        clSetKernelArg(cl_ctx.mating_kernel, 0, sizeof(cl_mem), &ctx.male_population);
        clSetKernelArg(cl_ctx.mating_kernel, 1, sizeof(cl_mem), &ctx.female_population);
        clSetKernelArg(cl_ctx.mating_kernel, 2, sizeof(cl_mem), &offspring_population);
        clSetKernelArg(cl_ctx.mating_kernel, 3, sizeof(cl_mem), &offspring_fitness_buf);
        clSetKernelArg(cl_ctx.mating_kernel, 4, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.mating_kernel, 5, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.mating_kernel, 6, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.mating_kernel, 7, sizeof(cl_int), &half_pop);

        size_t mating_work_size = MFA_NUM_OFFSPRING / 2;
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.mating_kernel, 1, NULL, &mating_work_size, NULL, 0, NULL, &mating_event);
        check_cl_error(err, "clEnqueueNDRangeKernel mating_phase");
        clFinish(cl_ctx.queue);

        double mating_time = 0.0;
        if (mating_event) {
            err = clGetEventProfilingInfo(mating_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(mating_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    mating_time = (end - start) / 1e6;
                    prof.mating_time += mating_time;
                }
            }
            clReleaseEvent(mating_event);
        }

        // Mutation phase
        cl_event mutation_event;
        cl_mem mutant_population = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, MFA_NUM_MUTANTS * dim * sizeof(cl_float), NULL, &err);
        check_cl_error(err, "clCreateBuffer mutant_population");
        cl_mem mutant_fitness_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, MFA_NUM_MUTANTS * sizeof(cl_float), NULL, &err);
        check_cl_error(err, "clCreateBuffer mutant_fitness");

        clSetKernelArg(cl_ctx.mutation_kernel, 0, sizeof(cl_mem), &ctx.male_population);
        clSetKernelArg(cl_ctx.mutation_kernel, 1, sizeof(cl_mem), &ctx.female_population);
        clSetKernelArg(cl_ctx.mutation_kernel, 2, sizeof(cl_mem), &mutant_population);
        clSetKernelArg(cl_ctx.mutation_kernel, 3, sizeof(cl_mem), &mutant_fitness_buf);
        clSetKernelArg(cl_ctx.mutation_kernel, 4, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.mutation_kernel, 5, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.mutation_kernel, 6, sizeof(cl_float), &mutation_rate);
        clSetKernelArg(cl_ctx.mutation_kernel, 7, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.mutation_kernel, 8, sizeof(cl_int), &pop_size);

        size_t mutation_work_size = MFA_NUM_MUTANTS;
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.mutation_kernel, 1, NULL, &mutation_work_size, NULL, 0, NULL, &mutation_event);
        check_cl_error(err, "clEnqueueNDRangeKernel mutation_phase");
        clFinish(cl_ctx.queue);

        double mutation_time = 0.0;
        if (mutation_event) {
            err = clGetEventProfilingInfo(mutation_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(mutation_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    mutation_time = (end - start) / 1e6;
                    prof.mutation_time += mutation_time;
                }
            }
            clReleaseEvent(mutation_event);
        }

        // Evaluate populations on CPU
        float *male_positions_new = (float *)malloc(half_pop * dim * sizeof(float));
        float *female_positions_new = (float *)malloc(half_pop * dim * sizeof(float));
        float *offspring_positions = (float *)malloc(MFA_NUM_OFFSPRING * dim * sizeof(float));
        float *mutant_positions = (float *)malloc(MFA_NUM_MUTANTS * dim * sizeof(float));
        float *male_fitness_new = (float *)malloc(half_pop * sizeof(float));
        float *female_fitness_new = (float *)malloc(half_pop * sizeof(float));
        float *offspring_fitness_values = (float *)malloc(MFA_NUM_OFFSPRING * sizeof(float));
        float *mutant_fitness_values = (float *)malloc(MFA_NUM_MUTANTS * sizeof(float));

        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.male_population, CL_TRUE, 0, half_pop * dim * sizeof(cl_float), male_positions_new, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer male_population");
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.female_population, CL_TRUE, 0, half_pop * dim * sizeof(cl_float), female_positions_new, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer female_population");
        err = clEnqueueReadBuffer(cl_ctx.queue, offspring_population, CL_TRUE, 0, MFA_NUM_OFFSPRING * dim * sizeof(cl_float), offspring_positions, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer offspring_population");
        err = clEnqueueReadBuffer(cl_ctx.queue, mutant_population, CL_TRUE, 0, MFA_NUM_MUTANTS * dim * sizeof(cl_float), mutant_positions, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer mutant_population");

        for (int i = 0; i < half_pop; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = male_positions_new[i * dim + j];
            male_fitness_new[i] = (float)objective_function(pos);
            free(pos);
        }
        for (int i = 0; i < half_pop; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = female_positions_new[i * dim + j];
            female_fitness_new[i] = (float)objective_function(pos);
            free(pos);
        }
        for (int i = 0; i < MFA_NUM_OFFSPRING; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = offspring_positions[i * dim + j];
            offspring_fitness_values[i] = (float)objective_function(pos);
            free(pos);
        }
        for (int i = 0; i < MFA_NUM_MUTANTS; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = mutant_positions[i * dim + j];
            mutant_fitness_values[i] = (float)objective_function(pos);
            free(pos);
        }

        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.male_fitness, CL_TRUE, 0, half_pop * sizeof(cl_float), male_fitness_new, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer male_fitness");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.female_fitness, CL_TRUE, 0, half_pop * sizeof(cl_float), female_fitness_new, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer female_fitness");
        err = clEnqueueWriteBuffer(cl_ctx.queue, offspring_fitness_buf, CL_TRUE, 0, MFA_NUM_OFFSPRING * sizeof(cl_float), offspring_fitness_values, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer offspring_fitness");
        err = clEnqueueWriteBuffer(cl_ctx.queue, mutant_fitness_buf, CL_TRUE, 0, MFA_NUM_MUTANTS * sizeof(cl_float), mutant_fitness_values, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer mutant_fitness");

        // Update best solution
        float new_best = male_fitness_new[0];
        best_idx = 0;
        is_male = 1;
        for (int i = 1; i < half_pop; i++) {
            if (male_fitness_new[i] < new_best) {
                new_best = male_fitness_new[i];
                best_idx = i;
            }
        }
        for (int i = 0; i < half_pop; i++) {
            if (female_fitness_new[i] < new_best) {
                new_best = female_fitness_new[i];
                best_idx = i;
                is_male = 0;
            }
        }
        for (int i = 0; i < MFA_NUM_OFFSPRING; i++) {
            if (offspring_fitness_values[i] < new_best) {
                new_best = offspring_fitness_values[i];
                best_idx = i;
                is_male = -1;
            }
        }
        for (int i = 0; i < MFA_NUM_MUTANTS; i++) {
            if (mutant_fitness_values[i] < new_best) {
                new_best = mutant_fitness_values[i];
                best_idx = i;
                is_male = -2;
            }
        }

        if (new_best < ctx.best_fitness) {
            ctx.best_fitness = new_best;
            float *new_best_pos = (float *)malloc(dim * sizeof(float));
            if (is_male == 1) {
                for (int j = 0; j < dim; j++) new_best_pos[j] = male_positions_new[best_idx * dim + j];
            } else if (is_male == 0) {
                for (int j = 0; j < dim; j++) new_best_pos[j] = female_positions_new[best_idx * dim + j];
            } else if (is_male == -1) {
                for (int j = 0; j < dim; j++) new_best_pos[j] = offspring_positions[best_idx * dim + j];
            } else {
                for (int j = 0; j < dim; j++) new_best_pos[j] = mutant_positions[best_idx * dim + j];
            }
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), new_best_pos, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer best_position");
            err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &new_best, 0, NULL, NULL);
            check_cl_error(err, "clEnqueueWriteBuffer best_fitness");
            free(new_best_pos);
        }

        // Sort and select (on CPU for simplicity)
        typedef struct { float fitness; float *position; int is_male; } Solution;
        Solution *temp_pop = (Solution *)malloc(pop_size * sizeof(Solution));
        for (int i = 0; i < half_pop; i++) {
            temp_pop[i].fitness = male_fitness_new[i];
            temp_pop[i].position = &male_positions_new[i * dim];
            temp_pop[i].is_male = 1;
            temp_pop[i + half_pop].fitness = female_fitness_new[i];
            temp_pop[i + half_pop].position = &female_positions_new[i * dim];
            temp_pop[i + half_pop].is_male = 0;
        }
        for (int i = 0; i < MFA_NUM_OFFSPRING; i++) {
            if (i < half_pop) {
                temp_pop[half_pop - 1 - i].fitness = offspring_fitness_values[i];
                temp_pop[half_pop - 1 - i].position = &offspring_positions[i * dim];
                temp_pop[half_pop - 1 - i].is_male = 1;
            } else if (i < pop_size) {
                temp_pop[pop_size - 1 - (i - half_pop)].fitness = offspring_fitness_values[i - half_pop];
                temp_pop[pop_size - 1 - (i - half_pop)].position = &offspring_positions[i * dim];
                temp_pop[pop_size - 1 - (i - half_pop)].is_male = 0;
            }
        }
        for (int i = 0; i < MFA_NUM_MUTANTS; i++) {
            temp_pop[pop_size - 1 - i].fitness = mutant_fitness_values[i];
            temp_pop[pop_size - 1 - i].position = &mutant_positions[i * dim];
            temp_pop[pop_size - 1 - i].is_male = (i % 2);
        }

        int compare_solutions(const void *a, const void *b) {
            float fa = ((Solution *)a)->fitness;
            float fb = ((Solution *)b)->fitness;
            return (fa > fb) - (fa < fb);
        }
        qsort(temp_pop, pop_size, sizeof(Solution), compare_solutions);

        float *male_positions_sorted = (float *)malloc(half_pop * dim * sizeof(float));
        float *female_positions_sorted = (float *)malloc(half_pop * dim * sizeof(float));
        for (int i = 0; i < half_pop; i++) {
            memcpy(&male_positions_sorted[i * dim], temp_pop[i].position, dim * sizeof(float));
            memcpy(&female_positions_sorted[i * dim], temp_pop[i + half_pop].position, dim * sizeof(float));
        }

        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.male_population, CL_TRUE, 0, half_pop * dim * sizeof(cl_float), male_positions_sorted, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer male_population");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.female_population, CL_TRUE, 0, half_pop * dim * sizeof(cl_float), female_positions_sorted, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer female_population");

        free(temp_pop);
        free(male_positions_new);
        free(female_positions_new);
        free(offspring_positions);
        free(mutant_positions);
        free(male_fitness_new);
        free(female_fitness_new);
        free(offspring_fitness_values);
        free(mutant_fitness_values);
        free(male_positions_sorted);
        free(female_positions_sorted);
        clReleaseMemObject(offspring_population);
        clReleaseMemObject(offspring_fitness_buf);
        clReleaseMemObject(mutant_population);
        clReleaseMemObject(mutant_fitness_buf);

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

        func_count += pop_size + MFA_NUM_OFFSPRING + MFA_NUM_MUTANTS;

        // Update parameters
        inertia_weight *= MFA_INERTIA_DAMP;
        nuptial_dance *= MFA_DANCE_DAMP;
        random_flight *= MFA_FLIGHT_DAMP;

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.best_fitness);
        printf("Profiling (ms): Update Males = %.3f, Update Females = %.3f, Mating = %.3f, Mutation = %.3f\n",
               update_males_time, update_females_time, mating_time, mutation_time);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_pop_time);
    printf("Update Males: %.3f\n", prof.update_males_time);
    printf("Update Females: %.3f\n", prof.update_females_time);
    printf("Mating Phase: %.3f\n", prof.mating_time);
    printf("Mutation Phase: %.3f\n", prof.mutation_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Update Males: %.3f\n", prof.update_males_time / prof.count);
        printf("Update Females: %.3f\n", prof.update_females_time / prof.count);
        printf("Mating Phase: %.3f\n", prof.mating_time / prof.count);
        printf("Mutation Phase: %.3f\n", prof.mutation_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Copy best solution to optimizer
    float *final_best_pos = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), final_best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer final best_pos");
    opt->best_solution.fitness = ctx.best_fitness;
    for (int i = 0; i < dim; i++) opt->best_solution.position[i] = final_best_pos[i];
    free(final_best_pos);

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
    MfA_cleanup_context(&ctx, &cl_ctx);
    MfA_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
